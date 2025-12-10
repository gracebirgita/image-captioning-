import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import pickle
import re, string
import keras
import collections

import os, json, time, cv2
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image

D_MODEL = 512
EMBED_DIM = 512
NUM_HEADS=2
DFF= 2048 # =ff dim = 4*512
NUM_LAYERS=2
MAX_CAP_LEN = 20
VOCAB_SIZE=10000
BATCH_SIZE=64
LR=1e-4 #standarny

def create_padding_mask(seq):
    return tf.cast(seq != 0, tf.bool)   # (batch, seq_len)
# ENCODER BLOCK
# seq fitur 3D tensor, self attention,
@keras.saving.register_keras_serializable()
class TransformerEncoderBlock(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads

        # key per head
        key_dim = embed_dim // num_heads

        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim = key_dim, dropout=0.0
        )

        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2=layers.LayerNormalization()

        self.dense_1 = layers.Dense(embed_dim, activation='relu') # perlebar dim
        self.dense_2 = layers.Dense(embed_dim) # persempit kembali

    def call(self, inputs, training=None, mask=None):
        attn_out = self.attention_1(query=inputs, value=inputs, key=inputs, 
                                    attention_mask=None,
                                    training=training)
        skip_1 = self.layernorm_1(inputs + attn_out) # residual conn1 + norm
        dense_out1 = self.dense_1(skip_1) # ffn
        dense_out2 = self.dense_2(dense_out1)
        out_1 = self.layernorm_2(skip_1+dense_out2) # residual conn 2 + norm
        return out_1

    def get_config(self):
        config=super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "dense_dim": self.dense_dim,
            "num_heads":self.num_heads,
        })
        return config
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
def build_visual_encoder_all(d_model=D_MODEL, dense_dim=256, num_heads=8):
    # 1. input layer
    inp = layers.Input(shape=(224,224,3), dtype=(tf.float32))
    # 2. base efficientnet pretrain
    base =EfficientNetB0(include_top=False, weights='imagenet') # none, h,w,c
    base.trainable=False # tdk include MLP model (hanya extract static)
    feat = base(inp) # shape(batch, Height, Weight, C)
    # 3. reshape & projeksi dense
    # flatten dimensi spasial -> sequence
    C =feat.shape[-1] # static
    # flatten spatial dim -> (B, H*W, C)
    seq = layers.Reshape((-1,C))(feat) # B, H*W, C
    # projection ke dim model
    proj = layers.Dense(d_model, name='proj_visual')(seq) #b,49,256
    # 4. encoder block (encoder layer + visual encoder)
    enc_output = TransformerEncoderBlock(
        embed_dim = d_model,
        dense_dim = dense_dim,
        num_heads = num_heads
    )(proj) # b, 49, 256
    model=tf.keras.Model(inputs=inp, outputs=enc_output, name='complete_visual_encoder')
    return model


class PositionalEmbedding(layers.Layer):
    def __init__(self, max_seq_len, vocab_size, d_model, **kwargs):
        super().__init__(**kwargs)
        self.token_emb = layers.Embedding(
            input_dim=vocab_size, output_dim=d_model
        )
        self.pos_emb = layers.Embedding(
            input_dim=max_seq_len, output_dim=d_model
        )
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embed_scale = tf.math.sqrt(tf.cast(d_model, tf.float32))

    def call(self, inputs):
        # input = batch, seq (32, 19)
        seq_len = tf.shape(inputs)[-1]  
        positions = tf.range(start=0, limit=seq_len, dtype=tf.int32)
        embedded_positions = self.pos_emb(positions)[tf.newaxis, ...] # seqlen,dmodel -> 1,seqlen, dmodel
        embedded_tokens = self.token_emb(inputs) * self.embed_scale
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)
# layer DECODER
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads=num_heads
        self.dff = dff
        self.rate = rate

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        
        head_dim = d_model//num_heads
        # tf.print("MHA1 expected mask shape (B, 1, Q, K) =", 
        #  [tf.shape(x)[0], 1, tf.shape(x)[1], tf.shape(x)[1]])

        # masked self attention (hanya lihat token sblmny)
        self.mha1=layers.MultiHeadAttention(num_heads=num_heads, 
                                            key_dim=head_dim, dropout=rate)
        
        # tf.print("MHA2 expected mask shape (B, 1, Q, K) =",
        #  [tf.shape(x)[0], 1, tf.shape(x)[1], tf.shape(enc_output)[1]])
        # encoder decoder cross attention(key=value=encoder output, query=output decoder, output global_region+bovw lihat isi gbr)
        self.mha2=layers.MultiHeadAttention(num_heads=num_heads, 
                                            key_dim=head_dim, dropout=rate)
        
        # ffn
        self.ffn = tf.keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model)    
        ])

        # layer norm
        self.layernorm1=layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2=layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3=layers.LayerNormalization(epsilon=1e-6)

        # dropout
        self.dropout1=layers.Dropout(rate)
        self.dropout2=layers.Dropout(rate)
        self.dropout3=layers.Dropout(rate)
        self.supports_masking=True

    
    def call(self,inputs, enc_output,training=False, self_mask=None, cross_mask=None, debug=False, return_attention=False):
        # 1. masked self-attention
        # x=caption embeding (masking spy decoder tdk lihat next token)

        attn1_out, attn1_scores = self.mha1(
            query=inputs, key=inputs, value=inputs,
            attention_mask=self_mask, #b, q, k
            training=training,
            return_attention_scores=True
        )
        attn1_out = self.dropout1(attn1_out, training=training)
        out1 = self.layernorm1(inputs + attn1_out)

        # 2. Cross-attention -- abaikan pad sj
        attn2_out, attn2_scores = self.mha2(
            query=out1,
            key=enc_output,
            value=enc_output,
            attention_mask=cross_mask, # enc padding mask (b,q,k encoder)
            training=training,
            return_attention_scores=True
        )
        attn2_out = self.dropout2(attn2_out, training=training)
        out2 = self.layernorm2(out1 + attn2_out)
       
        # 3. feed forward network
        ffn_output = self.ffn(out2)
        ffn_output=self.dropout3(ffn_output, training=training)
        out3=self.layernorm3(out2+ffn_output)
        # tf.print("x shape:", tf.shape(x))
        # tf.print("attn1_out:", tf.shape(attn1_out))
        if debug:
            return {
                "attn1": out1,
                "attn2": out2,
                "ffn": out3,
                
                "attn1_scores": attn1_scores,
                "attn2_scores": attn2_scores
            }
        return out3
    
    def get_config(self):
        config= super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "rate":self.rate, #dropout rate
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
# TRANSFORMER DECODER
class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, 
                 max_seq_len, rate=0.1, **kwargs):
        super().__init__(**kwargs)

        # token embedding (token ID -> vektor) -> positional embedding
        self.num_layers=num_layers
        self.num_heads=num_heads
        self.dff=dff

        self.d_model=d_model
        self.vocab_size=vocab_size
        self.max_seq_len=max_seq_len
        self.rate=rate

        # self.embedding = layers.Embedding(vocab_size, d_model)
        # self.pos_embedding = layers.Embedding(max_seq_len, d_model)
        self.embedding = PositionalEmbedding(
            max_seq_len=max_seq_len,
            vocab_size=vocab_size,
            d_model=d_model
        )
        self.dec_layers = [
            DecoderLayer(d_model, num_heads, dff, rate)
            for _ in range(num_layers)
        ]

        self.final_layernorm = layers.LayerNormalization(epsilon=1e-6)
        self.final_dense = layers.Dense(vocab_size)
        self.dropout = layers.Dropout(rate)
        self.supports_masking=True

    def get_causal_mask(self, seq_len):
        # 1 where allowed (i >= j), 0 future
        i = tf.range(seq_len)[:, None]
        j = tf.range(seq_len)[None, :]
        mask = tf.cast(i >= j, dtype=tf.int32)  # (seq, seq)
        return tf.reshape(mask, (1, seq_len, seq_len))  # (1, seq, seq)

    def call(self, inputs, enc_output, training=False, mask=None, enc_mask=None, debug=False):        
        # ver positional encode (class)
        # x=input decoder (batch, seq len)
        # enc output = output enc (32,49,256)

        # awal = batch,seq
        x = self.embedding(inputs) # B,seqlen, d model
        x = self.dropout(x, training=training)

        seq_len = tf.shape(inputs)[1]

        causal_mask = tf.cast(self.get_causal_mask(seq_len), tf.bool)#1,s,s

        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], tf.bool) #batch,1,seq
            # dec_padding_for_min = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)    # (batch, 1, seq)
            # # but we need shapes (batch, seq, seq) -> create dec_padding_for_min tiled:
            dec_padding_full = tf.tile(padding_mask, [1, seq_len, 1])     # (batch, seq, seq)
            combined_mask = tf.logical_and(dec_padding_full, causal_mask)   #b,s,s
        else:
            combined_mask= causal_mask  

        enc_attention_mask = None
        if enc_mask is not None:
            # enc_mask: (batch, enc_len) True=tdk pad
            # MultiHeadAttention (batch, 1, 1, enc_len) boolean- broadcast(batch, num_heads, query_len, enc_len)
            enc_attention_mask = enc_mask[:, None, :]   # (B, 1, enc_len) --akan expand ke (B, S, enc_len)

            # enc_attention_mask = tf.cast(enc_mask[:, tf.newaxis, tf.newaxis, :], tf.bool)

        # print(ross mask:", enc_att"self mask :", combined_mask.shape)
        # print("cention_mask.shape if enc_attention_mask is not None else None)

        # self attention (padding mask & causal caption)
        # debug
        if debug:
            attention_weights={}
        else:
            attention_weights=None

        x_in = x
        # loop dgn decoder layers yg sama
        for i, dec_layer in enumerate(self.dec_layers):
            if debug:
                out = dec_layer(
                    x_in,
                    enc_output,
                    training=training,
                    self_mask=combined_mask,
                    cross_mask=enc_attention_mask,
                    debug=True,
                )
                # simpan per-layer
                attention_weights[f"decoder_layer_{i}"] = out
                x_in = out["ffn"]
            else:
                x_in = dec_layer(inputs=x_in, 
                            enc_output=enc_output, 
                            training=training, 
                            self_mask=combined_mask,
                            cross_mask=enc_attention_mask
                            )
            
        # debug output tiap layer

        # logits = self.final_layer(x)
        # hidden state jd dist prob token
        x_in= self.final_layernorm(x_in) # normalisasi
        logits = self.final_dense(x_in) # B, seq_len, vocab_size

        if debug:
            return logits, attention_weights
        return logits

    def get_config(self):
        config=super().get_config()
        config.update({
            "num_layers": self.num_layers,
            "d_model": self.d_model,
            "num_heads":self.num_heads,
            "dff":self.dff,
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "rate": self.rate
        })
        return config
    @classmethod
    def from_config(cls, config):
        return cls(**config)

# WRAPPER : CAPTION TRAINER
@keras.saving.register_keras_serializable()
class CaptionTrainer(tf.keras.Model):
    def __init__(self, vocab_size, max_seq_len, d_model, num_heads, dff, num_layers, rate=0.1, num_captions_per_image=1, img_aug=None,**kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.num_layers = num_layers
        self.rate = rate
        self.num_captions_per_image = num_captions_per_image
        self.img_aug=img_aug

        # def build_visual_encoder_all(d_model=D_MODEL, dense_dim=256, num_heads=8):
        self.encoder = build_visual_encoder_all(d_model=self.d_model)          # efficient net encoder
        self.decoder = TransformerDecoder(
                        num_layers=self.num_layers, 
                        d_model=self.d_model, 
                        num_heads=self.num_heads, 
                        dff=self.dff, 
                        vocab_size=self.vocab_size, 
                        max_seq_len=self.max_seq_len, 
                        rate=rate) 

        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, 
            reduction=None
        )
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.acc_tracker = tf.keras.metrics.Mean(name="accuracy")
    
    
    def call(self, inputs, training=False):
        # inputs: tuple (img, caption_inp)
        imgs, captions_inp = inputs # dec in

        enc_out = self.encoder(imgs, training=training)
        dec_pad_mask = create_padding_mask(captions_inp) #batch,seqlen

        
        logits = self.decoder(inputs=captions_inp, 
                              enc_output=enc_out, 
                              mask=dec_pad_mask,
                              enc_mask=None,
                              training=training)
        return logits

    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "num_layers": self.num_layers,
            "rate": self.rate,
            "num_captions_per_image": self.num_captions_per_image,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def calculate_loss(self, y_true, y_pred, mask):
        loss = self.loss_fn(y_true, y_pred)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)


    def calculate_accuracy(self, y_true, y_pred, mask):
        y_true = tf.cast(y_true, tf.int32)

        pred_ids = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        matches = tf.equal(y_true, pred_ids)

        matches = tf.logical_and(tf.cast(mask, tf.bool), matches)
        matches = tf.cast(matches, tf.float32)
        mask = tf.cast(mask, tf.float32)

        return tf.math.divide_no_nan(tf.reduce_sum(matches), tf.reduce_sum(mask))
    

    def compute_loss_and_acc(self, img_embed, caption_inp, caption_true, training):
        # loss_mask = tf.cast(tf.not_equal(caption_true, 0), tf.float32)
        # seq_len = tf.shape(caption_inp)[1]

        dec_pad_mask = create_padding_mask(caption_inp)
        loss_mask = tf.cast(dec_pad_mask, tf.float32)
        logits = self.decoder(inputs=caption_inp, 
                              enc_output=img_embed, 
                              mask=dec_pad_mask,
                              enc_mask=None,
                              training=training)
        
        loss = self.calculate_loss(caption_true, logits, loss_mask) 
        acc  = self.calculate_accuracy(caption_true, logits, loss_mask)
        return loss, acc


    def train_step(self, data):
        (batch_img, dec_in_batch), dec_true = data
        train_vars = self.trainable_variables

        with tf.GradientTape() as tape:
            enc_out = self.encoder(batch_img, training=False) # b,49,dim model
            loss, acc = self.compute_loss_and_acc(
                enc_out, 
                dec_in_batch, # Input Decoder
                dec_true,  # Target Label (Ground Truth)
                training=True
            )

        grads = tape.gradient(loss, train_vars)
        self.optimizer.apply_gradients(zip(grads, train_vars))

        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)
        return {
            "loss": self.loss_tracker.result(),
            "accuracy": self.acc_tracker.result()
        }

    def test_step(self, data):
        # (batch_img, dec_in_batch), dec_out_batch = data
        (batch_img, dec_in_batch), dec_true = data

        # 1. Encode image/img embedding -- (frozen)
        enc_out = self.encoder(batch_img, training=False) 
        loss, acc = self.compute_loss_and_acc(
            enc_out, 
            dec_in_batch, # Input Decoder
            dec_true,  # Target Label (Ground Truth)
            training=False
        )

        # Update the trackers
        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)
        return {
            "loss": self.loss_tracker.result(),
            "accuracy": self.acc_tracker.result()
        }
    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_tracker]

# warmuplinear
@keras.saving.register_keras_serializable()
class WarmUpLinear(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, warmup_steps, final_lr, decay_type="linear"):
        super().__init__()
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.final_lr = final_lr
        self.decay_type = decay_type

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        # Warmup
        warmup_lr = self.initial_lr * (step / self.warmup_steps)
        # Linear decay
        linear_lr = tf.linspace(self.initial_lr, self.final_lr, 10000)[tf.cast(step, tf.int32) % 10000]

        # Choose based on step
        return tf.where(step < self.warmup_steps, warmup_lr, linear_lr)

    def get_config(self):
        return {
            "initial_lr": self.initial_lr,
            "warmup_steps": self.warmup_steps,
            "final_lr": self.final_lr,
            "decay_type": self.decay_type,
        }