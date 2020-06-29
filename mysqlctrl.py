import traceback
import pymysql
import random
import json
import datetime


#context cursor
_cursor = None
_showSQL = False


def contextCursor():
	return _cursor


class JsonDateTimeEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, datetime.datetime):
			return obj.strftime('%Y-%m-%d %H:%M:%S')
		elif isinstance(obj, datetime.date):
			return obj.strftime('%Y-%m-%d')
		else:
			return json.JSONEncoder.default(self, obj)

#convert have MappedData to standard data structure
def convertToSTD(data):
	if isinstance(data, list):
		return list(map(lambda x: convertToSTD(x), data))
	elif isinstance(data, MappedData):
		return data.getMappedData()
	else:
		return data


def sqlfilter(s):
	s = s.replace("\\", "\\\\")
	s = s.replace("'", "\\'")
	return s


def sqllist(lst):
	if lst is None or len(lst) < 1:
		return None

	s = ""
	for i in range(len(lst)):

		item_format = "'{}'".format(sqlfilter(str(lst[i])))
		if i == 0:
			s = item_format
		else:
			s += "," + item_format
	return "({})".format(s)


def filter(d):
	if isinstance(d, str):
		return "'{}'".format(sqlfilter(d))
	elif isinstance(d, list):
		return sqllist(d)
	elif d is None:
		return 'null'
	else:
		return str(d)

def sqlformat(sql, **args):
	newmap = {}
	for key in args:
		newmap[key] = filter(args[key])
	return sql.format(**newmap)

def noneOr(val, orval):
	return val if val is not None else orval


class MappedData:

	def __init__(self, **kwargs):

		if len(kwargs) == 0:
			self.data = None
			self.description = None
			self.colname = None
			self.packaged = False
		else:
			self.data = list(kwargs["data"]) if kwargs["data"] is not None else None
			self.description = kwargs["description"]
			self.colname = [item[0] for item in self.description] if self.description is not None else None
			self.packaged = False

	def __iter__(self):
		return self.mpdata

	def getMappedData(self):
		if self.packaged:
			return self.mpdata

		self.mpdata = self.mapped()
		self.packaged = True
		return self.mpdata

	def empty(self):
		return self.size == 0

	def extend(self, md):
		if not isinstance(md, MappedData):
			raise RuntimeError(f"unsupport type {md.__class__.__name__}, except MappedData")

		self.packaged = False
		if self.data is None:
			self.data = md.data
			self.description = md.description
			self.colname = md.colname
		else:
			self.data.extend(md.data)

	def mapped(self, one=False):
		data = self.data
		if data is None or self.description is None:
			return None

		if one and self.size > 1:
			raise "self.size > 1"

		newdata = {}
		for i in range(len(self.description)):
			name = self.description[i][0]
			if one:
				items = data[0][i] if data is not None and len(data) > 0 else None
			else:
				items = [data[j][i] for j in range(len(data))]
			newdata[name] = items
		return newdata

	def select(self, indces):
		if self.empty():
			return None

		newdata = []
		for ind in indces:
			newdata.append(self.data[ind])
		return MappedData(data=newdata, description=self.description)

	def choice(self, n):
		if self.empty():
			return None

		n = max(0, min(self.size-1, n))
		indces = list(range(self.size))
		random.shuffle(indces)
		return self.select(indces[:n])

	@property
	def size(self):
		return len(self.data) if self.data is not None else 0

	def __len__(self):
		return self.size

	def __getitem__(self, item):
		if isinstance(item, str):
			return self.getMappedData()[item]
		elif isinstance(item, slice):
			start = noneOr(item.start, 0)
			stop = noneOr(item.stop, self.size)
			step = noneOr(item.step, 1)
			return self.select(list(range(start, stop, step)))
		return self.mapped([self.data[item]], one=True)

	def __str__(self):
		body = str(self.getMappedData())
		return "MappedData, size: {}, data: {}".format(self.size, body)

	__repr__ = __str__


class Cursor:
	def __init__(self, db):
		self.cursor = db.cursor()
		self.db = db

	def __del__(self):
		self.cursor.close()

	def lastid(self):
		return self.cursor.lastrowid

	def exec(self, sql, **args):
		
	#	self.db.ping(reconnect=True)
		sql = sqlformat(sql, **args)
		print(sql)

		if _showSQL:
			print("exec sql[{}]".format(sql))

		self.cursor.execute(sql)

	def mapdata(self, data):
		return MappedData(data=data, description=self.cursor.description)

	def fetch(self, sql, rows = -1, **args):
		self.exec(sql, **args)

		if rows == -1:
			return self.mapdata(self.cursor.fetchall())
		else:
			return self.mapdata(self.cursor.fetchmany(rows))


class DB:

	def __init__(self, ip, name, pwd, db, port=3306):
		self.db = pymysql.connect(ip, name, pwd, db, charset="utf8", port=port)
		self.cursor = Cursor(self.db)

		global _cursor
		_cursor = self.cursor

	def __enter__(self):
		self.db.ping(reconnect=True)
		self.db.begin()
		return self.cursor

	def __exit__(self, exc_type, exc_val, exc_tb):
		global _cursor
		_cursor = None
		if exc_tb is not None:
			self.db.rollback()
			traceback.print_tb(exc_tb)
		else:
			self.db.commit()
