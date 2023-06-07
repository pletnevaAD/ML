from matplotlib import pyplot as plt
import seaborn as sns


def get_plots(array):
    fig = plt.figure()

    ax_1 = fig.add_subplot(1, 2, 1)
    ax_2 = fig.add_subplot(1, 2, 2)

    sns.histplot(
        ax=ax_1,
        data=array,
        kde=True,
        line_kws={"lw": 3})
    sns.ecdfplot(
        ax=ax_2,
        data=array)

    ax_1.set_title('kdeplot',
                   fontsize=15)
    ax_2.set_title('ecdfplot',
                   fontsize=15)

    fig.set_figwidth(14)
    fig.set_figheight(7)

    plt.show()
