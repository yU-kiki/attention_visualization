import matplotlib
from flask import Flask, render_template, url_for, request, redirect
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from transformers import GPT2Model, GPT2Tokenizer
import torch
import numpy as np
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
    inputs = tokenizer.encode(text, return_tensors='pt')
    outputs = model(inputs)
    all_layer_attentions = outputs.attentions

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Layer')
    ax.set_zlabel('Attention Score')
    tokens = tokenizer.tokenize(text)

    for i, layer_attention in enumerate(all_layer_attentions):
        layer_attention = layer_attention[0, 0, :, :].detach().numpy()
        X = np.array([i]*layer_attention.shape[0])
        Y = np.arange(layer_attention.shape[1])
        X, Y = np.meshgrid(Y, X)
        Z = layer_attention
        ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)

    ax.set_xticks(np.arange(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90)

    plt.savefig(os.path.join('static', 'images', 'plot.png'))

    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=True)
