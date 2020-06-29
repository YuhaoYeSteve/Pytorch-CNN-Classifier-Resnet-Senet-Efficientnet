from mysqlctrl import *
from logctrl import getLogger

logger = getLogger(__name__, "logs/all.log")

host = "10.1.1.231"
port = 10003
username = "root"
password = "123456"
database = "deepblue_bigbluecloud_release"
# database = "deepblue_bigbluecloud"
db = DB(host, username, password, database, port=port)

def restore_model():
    with db as cursor:
        row = cursor.exec(
            "update bis_bigblue_goods_group set classify_train_status='Created' where classify_train_status='Dispatch' or classify_train_status='Training'"
        )

def get_model_id():
    with db as cursor:
        row = cursor.fetch(
            "select id as modelId, `name` as modelName from bis_bigblue_goods_group bbgg where bbgg.classify_train_status='Created' or bbgg.classify_train_status='Dispatch' or classify_train_status='Training' ",
            row=1)
        if row.empty():
            return None
        item = row.mapped(False)
        return item["modelId"][0]

def get_model_goods(model_id):
    with db as cursor:
        row = cursor.fetch(
            "select id, name, barcode from bis_bigblue_goods_group_goods_relation left join inn_bigblue_goods on inn_bigblue_goods.id = bis_bigblue_goods_group_goods_relation.inn_bigblue_goods_id where bis_bigblue_goods_group_id = {model_id}",
            model_id=model_id)
        if row.empty():
            return None
        item = row.mapped(False)
        return item

def update_model_status(model_id, status):
    with db as cursor:
        row = cursor.exec(
            "update bis_bigblue_goods_group set classify_train_status={status} where id = {model_id}",
            status=status,
            model_id=model_id)
        logger.info(status + " classify train " + str(model_id))

def dispatch_model_status(model_id):
    with db as cursor:
        row = cursor.exec(
            "update bis_bigblue_goods_group set classify_dispath_time=now(), classify_train_status='Dispatch' where id = {model_id}",
            model_id=model_id)
        logger.info("dispatch classify train " + str(model_id))

def finish_model_status(model_id, train_status, trainResult, group_status):
    with db as cursor:
        row = cursor.exec(
            "update bis_bigblue_goods_group set classify_finish_time=now(), classify_result={trainResult}, classify_train_status={train_status}, status={group_status} where id={model_id}",
            train_status=train_status,
            trainResult=trainResult,
            model_id=model_id,
            group_status=group_status)
        logger.info("finish classify train " + str(model_id))
