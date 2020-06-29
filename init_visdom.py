from visdom import Visdom

def init_visdom_(window_name="CenterNet_train_ganggan"):
    # 在服务器命令行里输出 python -m visdom.server -p 8100 启动visdom服务
    vis = Visdom("http://localhost", port=8098, env=window_name)
    return vis