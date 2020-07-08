import datetime


# Write Log into txt
def save_log(txt_path, string_contant):
    with open(txt_path, "a") as f:
        nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S:')
        string_contant = nowTime + string_contant + "\n"
        f.writelines(string_contant)


# 跟新打印loss的间隔
def update_print_loss_interval(config, length_of_dataset):
    if (length_of_dataset / config.batch_size) < 2 * 10:
        config.print_loss_interval = 1
        config.print_loss_remainder = 0
    elif 2 * 10 <= (length_of_dataset / config.batch_size) < 2 * 100:
        config.print_loss_interval = 10
        config.print_loss_remainder = 9
    elif 2 * 100 <= (length_of_dataset / config.batch_size) < 2 * 1000:
        config.print_loss_interval = 100
        config.print_loss_remainder = 99
    elif 2 * 1000 <= (length_of_dataset / config.batch_size) < 2 * 10000:
        config.print_loss_interval = 1000
        config.print_loss_remainder = 999
    elif (length_of_dataset / config.batch_size) >= 2 * 10000:
        config.print_loss_interval = 10000
        config.print_loss_remainder = 9999


if __name__ == "__main__":
    log_path = "./1.txt"
    train_info = "loss: {}".format(round(0.2222123123124, 6))
    save_log(log_path, train_info)
