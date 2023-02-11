{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "private_outputs": true,
      "provenance": [],
      "authorship_tag": "ABX9TyN3JDfYoB5EH3+VREgqiko5",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/sukritis312/anamoly-detection/blob/main/anamoly_detection.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install tensorflow==2.8\n",
        "!pip install keras"
      ],
      "metadata": {
        "id": "wYV3iTE-sRPW"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install gensim\n",
        "!pip install python-Levenshtein"
      ],
      "metadata": {
        "id": "pZ6hColTsjTj"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "!wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/Advanced_ML_anomaly_detection_L3/DataSets.zip"
      ],
      "metadata": {
        "id": "GTvqLRKrtflN"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install np_utils"
      ],
      "metadata": {
        "id": "2uH8EV0zvF-U"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install --upgrade numpy"
      ],
      "metadata": {
        "id": "4xmPNDlbvwYv"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "import tensorflow as tf\n",
        "import matplotlib.pyplot as plt\n",
        "import xml.etree.ElementTree as ElTree\n",
        "import re, h5py, itertools, math, glob, zipfile, os\n",
        "from sklearn.preprocessing import LabelEncoder, OneHotEncoder\n",
        "from sklearn.metrics import log_loss, auc, roc_curve\n",
        "from tensorflow.keras import layers\n",
        "from tensorflow.keras.layers import Masking,Activation\n",
        "from tensorflow.keras.layers import Dense, LSTM, Dropout, Embedding, TimeDistributed,Bidirectional\n",
        "from tensorflow.keras.models import Model, Sequential, load_model\n",
        "from tensorflow.keras.utils import to_categorical\n",
        "from tensorflow.python.client import device_lib\n",
        "from lxml import etree\n",
        "from gensim.models import Word2Vec\n",
        "\n",
        "# %matplotlib inline\n",
        "plt.rcParams['figure.figsize'] = (15, 5)\n",
        "plt.style.use('ggplot')\n",
        "seed = 42\n",
        "\n",
        "import warnings\n",
        "warnings.filterwarnings(action = \"ignore\")"
      ],
      "metadata": {
        "id": "AprcM1Crtih7"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#set precision value\n",
        "pd.set_option(\"precision\", 3)\n",
        "pd.options.display.float_format = '{:.3f}'.format"
      ],
      "metadata": {
        "id": "4B6-YVu5v8qS"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def plot_history(history):\n",
        "    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]\n",
        "    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]\n",
        "    acc_list = [s for s in history.history.keys() if 'accuracy' in s and 'val' not in s]\n",
        "    val_acc_list = [s for s in history.history.keys() if 'accuracy' in s and 'val' in s]\n",
        "    \n",
        "    plt.figure(figsize = (12, 5), dpi = 100)\n",
        "    COLOR = 'gray'\n",
        "    \n",
        "    plt.rc('legend', fontsize = 14)   # legend fontsize\n",
        "    plt.rc('figure', titlesize = 12)  # fontsize of the figure title\n",
        "        \n",
        "    if len(loss_list) == 0:\n",
        "        print('Loss is missing in history')\n",
        "        return \n",
        "    \n",
        "    ## As loss always exists\n",
        "    epochs = range(1, len(history.history[loss_list[0]]) + 1)\n",
        "    \n",
        "    ## Loss\n",
        "    plt.subplot(1, 2, 1)\n",
        "    plt.subplots_adjust(wspace = 2, hspace = 2)\n",
        "    plt.rcParams['text.color'] = 'black'\n",
        "    plt.rcParams['axes.titlecolor'] = 'black'\n",
        "    plt.rcParams['axes.labelcolor'] = COLOR\n",
        "    plt.rcParams['xtick.color'] = COLOR\n",
        "    plt.rcParams['ytick.color'] = COLOR\n",
        "    for l in loss_list:\n",
        "        plt.plot(epochs, history.history[l], 'b-o',\n",
        "                 label = 'Train (' + str(str(format(history.history[l][-1],'.4f'))+')'))\n",
        "    for l in val_loss_list:\n",
        "        plt.plot(epochs, history.history[l], 'g',\n",
        "                 label = 'Valid (' + str(str(format(history.history[l][-1],'.4f'))+')'))\n",
        "    \n",
        "    plt.title('Loss')\n",
        "    plt.xlabel('Epochs')\n",
        "    plt.legend(facecolor = 'gray', loc = 'best')\n",
        "    plt.grid(True)\n",
        "    plt.tight_layout()\n",
        "    \n",
        "    ## Accuracy\n",
        "    plt.subplot(1, 2, 2)\n",
        "    plt.subplots_adjust(wspace = 2, hspace = 2)\n",
        "    plt.rcParams['text.color'] = 'black'\n",
        "    plt.rcParams['axes.titlecolor'] = 'black'\n",
        "    plt.rcParams['axes.labelcolor'] = COLOR\n",
        "    plt.rcParams['xtick.color'] = COLOR\n",
        "    plt.rcParams['ytick.color'] = COLOR\n",
        "    for l in acc_list:\n",
        "        plt.plot(epochs, history.history[l], 'b-o',\n",
        "                 label = 'Train (' + str(format(history.history[l][-1],'.4f'))+')')\n",
        "    for l in val_acc_list:    \n",
        "        plt.plot(epochs, history.history[l], 'g',\n",
        "                 label = 'Valid (' + str(format(history.history[l][-1],'.4f'))+')')\n",
        "\n",
        "    plt.title('Accuracy')\n",
        "    plt.xlabel('Epochs')\n",
        "    plt.legend(facecolor = 'gray', loc = 'best')\n",
        "    plt.grid(True)\n",
        "    plt.tight_layout()\n",
        "    plt.show()\n",
        "\n",
        "class B_Generator(object):\n",
        "    def __init__(self, BZ, XX, YY, ohe):\n",
        "        self.BZ = BZ\n",
        "        self.n_b = int(math.floor(np.shape(XX)[0] / BZ))\n",
        "        self.b_index = [a * BZ for a in range(0, self.n_b)]\n",
        "        self.XX = XX\n",
        "        self.YY = YY\n",
        "        self.ohe = ohe\n",
        "        \n",
        "    def __iter__(self):\n",
        "        for var_0 in itertools.cycle(self.b_index):\n",
        "            YY = self.YY[var_0 : (var_0 + self.BZ)]\n",
        "            ohe_Y = self.ohe.transform(YY.reshape(len(YY), 1))\n",
        "            yield (self.XX[var_0 : (var_0 + self.BZ),], ohe_Y)"
      ],
      "metadata": {
        "id": "BD6Evlj1xIHm"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#reading the dataset\n",
        "number = 4\n",
        "\n",
        "if zipfile.is_zipfile('DataSets.zip'):\n",
        "  file_1 = zipfile.ZipFile('DataSets.zip', 'r')\n",
        "else:\n",
        "  print('Type file isn`t ZIP')\n",
        "\n",
        "name_dataset = file_1.namelist()[number]\n",
        "file_1.extract(name_dataset)\n",
        "print(\"File\", name_dataset, \"has been read\")"
      ],
      "metadata": {
        "id": "c5kRxTn6xKGW"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#parse the unzipped file of the dataset from XML\n",
        "tree_set = ElTree.parse(name_dataset)\n",
        "root_tree_set = tree_set.getroot()\n",
        "\n",
        "result = []\n",
        "var_1 = root_tree_set.items()[0][1][:-4]\n",
        "\n",
        "for item in root_tree_set.findall(var_1):\n",
        "    result.append({node.tag: node.text for node in item.getiterator()})\n"
      ],
      "metadata": {
        "id": "7sUrkaBixXim"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#convert it into data frame\n",
        "dSET = pd.DataFrame(result)\n",
        "dSET = dSET.drop(dSET.columns[[0]], axis = 1)\n",
        "dSET = dSET.drop_duplicates()\n",
        "\n",
        "dSET = dSET.sort_values('startDateTime')\n",
        "dSET['IPs_sequence'] = dSET['source'] + '_' + dSET['destination'] + '_' + dSET['startDateTime'].str[:13]\n",
        "\n",
        "dSET['res_port'] = np.where(dSET.destinationPort <= dSET.sourcePort,\n",
        "                            dSET['destinationPort'],\n",
        "                            dSET['sourcePort'])\n",
        "\n",
        "# rename some of the columns for more convenience\n",
        "dSET = dSET.rename(columns = {'totalSourceBytes': 'totSB',\n",
        "                        'totalDestinationBytes': 'totDB',\n",
        "                        'totalDestinationPackets': 'totDP',\n",
        "                        'totalSourcePackets': 'totSP',\n",
        "                        'sourcePayloadAsBase64': 'sourB64',\n",
        "                        'sourcePayloadAsUTF': 'sourUTF',\n",
        "                        'destinationPayloadAsBase64': 'destB64',\n",
        "                        'destinationPayloadAsUTF': 'destUTF',\n",
        "                        'direction': 'direct',\n",
        "                        'sourceTCPFlagsDescription': 'sourTCPFd',\n",
        "                        'destinationTCPFlagsDescription': 'destTCPFd',\n",
        "                        'protocolName': 'pName',\n",
        "                        'sourcePort': 'sPort',\n",
        "                        'destination': 'dest',\n",
        "                        'destinationPort': 'dPort'})\n",
        "print(\"Preparation process has been finished\")"
      ],
      "metadata": {
        "id": "hRw183oLx0OH"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#dataset size & feature names\n",
        "dSET.shape, dSET.columns"
      ],
      "metadata": {
        "id": "ygPgAWrqx3zh"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "dSET.head(5)"
      ],
      "metadata": {
        "id": "DrwXCv07ye5X"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "## Build the sets (keys and sequences) in hour slices\n",
        "print(\"Stage I. Keys building\\n\")\n",
        "key = dSET.groupby('IPs_sequence')[['Tag', 'res_port']].agg({\"Tag\": lambda var_2: \"%s\" % ','.join([var_3 for var_3 in var_2]),\n",
        "          \"res_port\" :lambda var_2: \"%s\" % ','.join([str(var_3) if int(var_3) < 10000 else \"10000\" for var_3 in var_2])})\n",
        "\n",
        "print(\"Unique keys:\\n\" + str(key.count()))\n",
        "attacks = [var_4.split(\",\") for var_4 in key.Tag.tolist()]\n",
        "sequences = [var_4.split(\",\") for var_4 in key.res_port.tolist()]"
      ],
      "metadata": {
        "id": "VqCaGkbYzRWL"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(\"Stage II. Label encoding\\n\")\n",
        "U_tokens = list(set([var_5 for var_6 in sequences for var_5 in var_6]))\n",
        "print(\"Number of unique tokens :\", len(U_tokens))\n",
        "LE = LabelEncoder().fit(U_tokens)\n",
        "sequences = [LE.transform(var_7).tolist() for var_7 in sequences]\n",
        "sequences = [[var_6 + 1 for var_6 in var_5] for var_5 in sequences]\n",
        "print(\"Number of sequences :\", len(sequences))\n",
        "sequence_attack = zip(attacks, sequences)"
      ],
      "metadata": {
        "id": "dcrCqoXZzlSP"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(\"Stage III. Sequences generating for the future model\\n\")\n",
        "var_8 = np.float32(0)\n",
        "len_sequence = 10\n",
        "print(\"Length of the primary sequence :\", len_sequence)\n",
        "seq_IDX, seq_X, seq_Y, seq_ATT = [], [], [], []\n",
        "for var_10, (var_11, var_12) in enumerate(sequence_attack):\n",
        "    sequence_1 = [np.float32(0)] * (len_sequence) + var_12\n",
        "    sequence_2 = [np.float32(0)] * (len_sequence) + var_11\n",
        "    for var_9 in range(len_sequence, len(sequence_1)):\n",
        "        sequence_3 = sequence_1[(var_9 - len_sequence):(var_9)]\n",
        "        var_14 = []\n",
        "        for var_13 in sequence_3:\n",
        "            try:\n",
        "                var_14.append(var_13)\n",
        "            except:\n",
        "                var_14.append(var_8)\n",
        "        seq_X.append(var_14)\n",
        "        seq_Y.append(sequence_1[var_9])\n",
        "        seq_IDX.append(var_10)\n",
        "        seq_ATT.append(sequence_2[var_9])\n",
        "print(\"Length of X & Y sets :\", len(seq_X))"
      ],
      "metadata": {
        "id": "_CBNZSov0G6c"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#One-hot-encoder initializing\n",
        "print(\"Stage IV. One-hot-encoder initializing\\n\")\n",
        "OHE = OneHotEncoder(sparse = False, categories = 'auto').fit(np.unique(seq_Y).reshape(-1, 1))\n",
        "\n",
        "X = np.array(seq_X)\n",
        "print(\"Dimensionality size of set X :\", X.shape)"
      ],
      "metadata": {
        "id": "oNfGbonC0li6"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#model building\n",
        "print(\"Stage V. Model building\\n\")\n",
        "drop_level = 0.35 # You can choose a drop level yourself\n",
        "N_neurons = 50   # You can choose the number of neurons yourself\n",
        "\n",
        "model = Sequential()\n",
        "model.add(layers.Embedding(output_dim = 100,\n",
        "                    input_dim = len(U_tokens) + 1,\n",
        "                    mask_zero = True))\n",
        "\n",
        "model.add(layers.Bidirectional(LSTM(N_neurons, return_sequences = True)))\n",
        "model.add(layers.Dropout(drop_level))\n",
        "\n",
        "model.add(layers.Bidirectional(LSTM(N_neurons, activation = \"relu\", return_sequences = False)))\n",
        "model.add(layers.Dropout(drop_level))\n",
        "\n",
        "model.add(layers.Dense(N_neurons, activation = \"linear\"))\n",
        "model.add(layers.Dropout(drop_level))\n",
        "\n",
        "model.add(layers.Dense(len(U_tokens), activation = \"softmax\"))\n",
        "\n",
        "model.summary()"
      ],
      "metadata": {
        "id": "PMrWZYtK01kc"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#model compiling and fitting\n",
        "print(\"Stage VI. Compile and fit the model\\n\")\n",
        "\n",
        "batch_size = 512  \n",
        "n_epochs = 10     \n",
        "\n",
        "optim = tf.keras.optimizers.Nadam()   # You can choose an optimizer yourself\n",
        "loss_f = tf.keras.metrics.categorical_crossentropy\n",
        "\n",
        "T_data = B_Generator(batch_size, np.asarray(X), np.asarray(seq_Y), OHE)\n",
        "\n",
        "model.compile(loss = loss_f,\n",
        "              optimizer = optim,\n",
        "              metrics = ['accuracy'])\n",
        "\n",
        "history = model.fit_generator(T_data.__iter__(),\n",
        "    steps_per_epoch = T_data.n_b,\n",
        "    epochs = n_epochs,\n",
        "    verbose = 1)"
      ],
      "metadata": {
        "id": "SJM-wlsw4D-i"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#results\n",
        "print(\"Stage VII. Results visualization\\n\")\n",
        "plot_history(history)"
      ],
      "metadata": {
        "id": "2qGP5e0t5WeF"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#model saving\n",
        "print(\"Stage VIII. Model saving & prediction checking\\n\")\n",
        "\n",
        "M_name = 'Detection_model'\n",
        "\n",
        "filepath = M_name + '.h5'\n",
        "tf.keras.models.save_model(model, filepath, include_optimizer = True, save_format = 'h5', overwrite = True)\n",
        "print(\"Size of the saved model :\", os.stat(filepath).st_size, \"bytes\")"
      ],
      "metadata": {
        "id": "gON_48E-5ryT"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model_L = tf.keras.models.load_model(filepath)\n",
        "predicts = model_L.predict(X, batch_size = batch_size)\n",
        "print(\"Dimensionality sizes of model predicts :\", predicts.shape, \"\\n\")\n",
        "print(\"Compare with length of X & Y sets :\\t\", len(seq_X), \"\\nand with number of tokens :\\t\\t\", len(U_tokens))"
      ],
      "metadata": {
        "id": "kKCPrAEh6BOx"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}