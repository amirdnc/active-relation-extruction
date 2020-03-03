
import numpy as np
import matplotlib.pyplot as plt

def read_predict_file(path, procces_sentance):
    lines = []
    with open(path, 'r') as f:
        l1, l2 = f.readline(), f.readline()
        f.readline()  # remove blank line
        while l1:
            lines.append(procces_sentance(l1, l2))  # first line is sentence, second is the emmbeddings
            l1, l2 = f.readline(), f.readline()
            f.readline()  #remove blank line
    return lines


def plt_heatmap(mat, label_dict: dict, title=''):
    fig, ax = plt.subplots()
    im = ax.imshow(mat)
    label_dict = [x[1] for x in sorted(label_dict.items())]
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(label_dict)))
    ax.set_yticks(np.arange(len(label_dict)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(label_dict)
    ax.set_yticklabels(label_dict)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(label_dict)):
        for j in range(len(label_dict)):
            text = ax.text(j, i,  '%.2f' % mat[i, j],
                           ha="center", va="center", color="w")

    ax.set_title(title)
    fig.tight_layout()
    fig.set_size_inches(20, 20, forward=True)
    plt.show()
