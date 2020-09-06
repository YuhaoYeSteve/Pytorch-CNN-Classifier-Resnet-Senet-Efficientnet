class BBox:
    """
    General Bounding Box class for storing box info
    :param label: must be [x, y] list format
    :param model: model can be "xywh"([top_left_x, top_left_y, width, height]) or "xyxy"([top_left_x, top_left_y, bottom_down_x, bottom_down_y]) or "point" [x, y]
    :param model: xyxy [top_left_x, top_left_y, bottom_down_x, bottom_down_y] python list
    :param model: xywh [top_left_x, top_left_y, width, height] python list
    :param model: cxcy [central_point_x, central_point_y] python list
    :param model: score float
    :return: numpy.ndarray(uint8)
    """
    def __init__(self, label, xyxy=[0, 0, 0, 0], xywh=[0, 0, 0, 0], cxcy=[0, 0], score=0, model="xywh"):

        self.label = label
        self.score = score
        self.x, self.y, self.r, self.b = xyxy
        self.cx, self.cy = cxcy

        # 避免出现rb小于xy的时候
        minx = min(self.x, self.r)
        maxx = max(self.x, self.r)
        miny = min(self.y, self.b)
        maxy = max(self.y, self.b)
        self.x, self.y, self.r, self.b = minx, miny, maxx, maxy

    # 调用print(bbox的时候会打印下面的信息)
    def __repr__(self):
        return f"(BBox[{self.label}]: x={self.x:.2f}, y={self.y:.2f}, r={self.r:.2f}, b={self.b:.2f}, width={self.width:.2f}, height={self.height:.2f})"

    # @property是装饰器, 把一个方法变成属性调用， 比如可以直接bbox.width
    @property
    def width(self):
        return self.r - self.x + 1

    @property
    def height(self):
        return self.b - self.y + 1

    @property
    def area(self):
        return self.width * self.height

    @property
    def box(self):
        return [self.x, self.y, self.r, self.b]

    @box.setter
    def box(self, newvalue):
        self.x, self.y, self.r, self.b = newvalue

    @property
    def xywh(self):
        return [self.x, self.y, self.width, self.height]

    @property
    def center(self):
        return [(self.x + self.r) * 0.5, (self.y + self.b) * 0.5]