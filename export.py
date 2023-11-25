import paddle
from model import RecurrentAttention

if __name__ == "__main__":
    model = RecurrentAttention(8, 1, 1, 1, 128, 126, 0.05, 256, 10)
    x = paddle.randn(shape=[4, 3, 320, 320])
    l_t_prev = paddle.randn(shape=[4, 2])
    h_t_prev = paddle.randn(shape=[4, 256])
    try:
        input_spec = list(
            paddle.static.InputSpec.from_tensor(paddle.to_tensor(t))
            for t in (x, l_t_prev, h_t_prev)
        )
        paddle.jit.save(model, input_spec=input_spec, path="./model")
        print("[JIT] paddle.jit.save successed.")
        exit(0)
    except Exception as e:
        print("[JIT] paddle.jit.save failed.")
        raise e
