import paddle
import sys
sys.path.append('..')
import model
import utils

if __name__ == "__main__":
    plot_dir = "../plots/"
    data_dir = "../data/"
    imgs = []
    paths = [data_dir + "./lenna.jpg", data_dir + "./cat.jpg"]
    for i in range(len(paths)):
        img = utils.img2array(paths[i], desired_size=[512, 512], expand=True)
        imgs.append(paddle.to_tensor(data=img))
    imgs = paddle.concat(x=imgs).transpose(perm=(0, 3, 1, 2))
    B, C, H, W = imgs.shape
    l_t_prev = paddle.empty(shape=[B, 2], dtype="float32").uniform_(min=-1, max=1)
    h_t_prev = paddle.zeros(shape=[B, 256])
    ram = model.RecurrentAttention(64, 3, 2, C, 128, 128, 0.11, 256, 10)
    h_t, l_t, _, _ = ram(imgs, l_t_prev, h_t_prev)
    assert h_t.shape == (B, 256)
    assert l_t.shape == (B, 2)
