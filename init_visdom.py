from visdom import Visdom

def init_visdom_(window_name="CenterNet_train_ganggan"):
    # 在服务器命令行里输出 python -m visdom.server -p 8100 启动visdom服务
    # Input "python -m visdom.server -p 8100" in terminel
    vis = Visdom("http://localhost", port=8100, env=window_name)
    return vis