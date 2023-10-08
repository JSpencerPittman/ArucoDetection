import cv2
import numpy as np


class Tag:
    def __init__(self, id, x, y, size, opacity, blur):
        self.m_id = id
        self.m_x = x
        self.m_y = y
        self.m_size = size
        self.m_opacity = opacity
        self.m_blur = blur

    def x(self):
        return self.m_x

    def y(self):
        return self.m_y

    def size(self):
        return self.m_size

    def draw_marker(self, image):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        marker = cv2.aruco.generateImageMarker(dictionary, self.m_id, self.m_size)
        marker = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        if self.m_blur > 0:
            marker = cv2.GaussianBlur(marker, (5, 5), self.m_blur)

        startRow = max(self.m_y - self.m_size // 2, 0)
        startCol = max(self.m_x - self.m_size // 2, 0)
        if startRow + self.m_size > image.shape[0]:
            startRow = image.shape[0] - self.m_size
        if startCol + self.m_size > image.shape[1]:
            startCol = image.shape[1] - self.m_size

        roi = image[startRow:startRow + self.m_size, startCol:startCol + self.m_size]
        cv2.addWeighted(roi, 1.0 - self.m_opacity/100.0, marker, self.m_opacity/100.0, 0, roi)

    def print(self):
        print(f"TAG: {self.m_id}")
        print(f"CENTER: {self.m_x}, {self.m_y}")
        print(f"SIZE: {self.m_size}")
        print(f"OPACITY & BLUR: {self.m_opacity} {self.m_blur}")

    @staticmethod
    def is_overlap(t1, t2):
        t1Radius = t1.m_size // 2
        t2Radius = t2.m_size // 2
        t1RangeX = (t1.m_x - t1Radius, t1.m_x + t1Radius)
        t2RangeX = (t2.m_x - t2Radius, t2.m_x + t2Radius)
        t1RangeY = (t1.m_y - t1Radius, t1.m_y + t1Radius)
        t2RangeY = (t2.m_y - t2Radius, t2.m_y + t2Radius)

        return Tag._range_overlap(t1RangeX, t2RangeX) and Tag._range_overlap(t1RangeY, t2RangeY)

    @staticmethod
    def _range_overlap(r1, r2):
        return r1[0] <= r2[1] and r1[1] >= r2[0]
