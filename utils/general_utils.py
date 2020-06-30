
import datetime

# Write Log into txt
def save_log(txt_path, string_contant):
    with open(txt_path, "a") as f:
        nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S:')
        string_contant = nowTime + string_contant + "\n"
        f.writelines(string_contant)

if __name__ == "__main__":
    log_path = "./1.txt"
    train_info = "loss: {}".format(round(0.2222123123124, 6))
    save_log(log_path, train_info)

