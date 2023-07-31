import matplotlib
from flask import Flask, render_template, url_for, request, redirect, Markup
from transformers import GPT2Model, GPT2Tokenizer, GPT2LMHeadModel
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

matplotlib.use('Agg')

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/generate_plot', methods=['POST'])
def generate_plot():
    model = GPT2Model.from_pretrained('gpt2', output_attentions=True)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    text = request.form['text']
    min_font_size = int(request.form['min_font_size'])
    max_font_size = int(request.form['max_font_size'])
    inputs = tokenizer.encode(text, return_tensors='pt')
    outputs = model(inputs)
    all_layer_attentions = outputs.attentions

    last_layer_attentions = all_layer_attentions[-1][0, 0, :, :].detach().numpy()
    tokens = tokenizer.tokenize(text)
    tokens = [token.replace('Ġ', '') for token in tokens]
    attention_text = generate_attention_text(
        tokens, last_layer_attentions, min_font_size, max_font_size)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Layer')
    ax.set_zlabel('Attention Score')

    for i, layer_attention in enumerate(all_layer_attentions):
        layer_attention = layer_attention[0, 0, :, :].detach().numpy()
        X = np.arange(layer_attention.shape[0])
        Y = np.arange(layer_attention.shape[1])
        X, Y = np.meshgrid(Y, X)
        Z = layer_attention
        ax.plot_surface(X, Y, Z, cmap='coolwarm')

    ax.set_xticks(np.arange(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90)

    plt.savefig(os.path.join('static', 'images', 'plot.png'))

    generation_model = GPT2LMHeadModel.from_pretrained('gpt2')
    generated_output = generation_model.generate(
        inputs, max_length=150, do_sample=True)
    generated_text = tokenizer.decode(
        generated_output[0], skip_special_tokens=True)

    return render_template('index.html', attention_text=attention_text, generated_text=generated_text)


def generate_attention_text(tokens, attentions, min_font_size=10, max_font_size=32):
    # Average attention scores over all heads
    avg_attention = np.mean(attentions, axis=0)
    min_val = np.min(avg_attention)
    max_val = np.max(avg_attention)
    norm_attentions = (avg_attention - min_val) / (max_val - min_val)

    # Scale font size by attention score. Range from min_font_size to max_font_size.
    scaled_font_sizes = min_font_size + \
        (max_font_size - min_font_size) * norm_attentions
    attention_text = ''
    for token, font_size, attention in zip(tokens, scaled_font_sizes, norm_attentions):
        color_intensity = int(255 * attention)
        color = f'rgb({color_intensity}, 0, {255-color_intensity})'
        attention_text += f'<span style="font-size: {font_size}pt; color: {color};">{token} </span>'

    return Markup(attention_text)



if __name__ == '__main__':
    app.run(debug=True)
