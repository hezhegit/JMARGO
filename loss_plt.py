import numpy as np
import matplotlib.pyplot as plt

def plot_loss(n1, n2, n3, save_dir):
    # 加载每个epoch的损失值
    y1 = np.load(save_dir + n1)
    y2 = np.load(save_dir + n2)
    y3 = np.load(save_dir + n3)

    # 计算x轴的值
    x = range(0, max(len(y1), len(y2), len(y3)))

    # 绘制三条线
    plt.plot(x[:len(y1)], y1, "r-", label='BPO')
    plt.plot(x[:len(y2)], y2, "g-", label='MFO')
    plt.plot(x[:len(y3)], y3, "b-", label='CCO')

    # 设置图表标题和坐标轴标签
    plt_title = 'MERGO Valid Loss'
    plt.title(plt_title)
    plt.xlabel('epoch')
    plt.ylabel('LOSS')

    # 显示图例
    plt.legend(loc='upper right')

    plt.show()
    # 保存图表
    # plt.savefig(save_dir + "Loss_Comparison.png")

# 调用函数示例
save_dir = 'models_weight/cafa3/'
n1 = 'BP/MERGO/checkpointEpoch_12.npy'
n2 = 'MF/MERGO/checkpointEpoch_12.npy'
n3 = 'CC/MERGO/checkpointEpoch_12.npy'

plot_loss(n1, n2, n3, save_dir)