from OpenGL.GL import *
from OpenGL.GLU import *
import torch

#Justin's big chessboard, squares 100mm
class ModelChessboard:
    def __init__(self, rows=8, cols=8, square_size=1.0):
        self.rows = rows
        self.cols = cols
        self.square_size = square_size
        self.width = cols * square_size
        self.height = rows * square_size
        self.midW = self.width / 2.0
        self.midH = self.height / 2.0

        self.cb_pts_2D_cart = torch.zeros(2, rows * cols)
        self.cb_pts_2D = torch.zeros(3, rows * cols)
        self.cb_pts_3D = torch.zeros(4, rows * cols)

        n_lines = (cols - 1) * rows + (rows - 1) * cols
        self.lines = torch.zeros(2, n_lines)

        ctr = 0

        line_ctr = 0
        for row in range(0, rows):
            for col in range(0, cols):
                x = col * square_size - self.midW
                y = row * square_size - self.midH
                self.cb_pts_2D_cart[0, ctr] = x
                self.cb_pts_2D_cart[1, ctr] = y

                if row < (rows - 1):
                    self.lines[0, line_ctr] = ctr
                    self.lines[1, line_ctr] = self.getIndex(row + 1, col)
                    line_ctr = line_ctr + 1

                if col < (cols - 1):
                    self.lines[0, line_ctr] = ctr
                    self.lines[1, line_ctr] = ctr + 1
                    line_ctr = line_ctr + 1
                    
                ctr = ctr + 1

        self.cb_pts_2D[:2, :] = self.cb_pts_2D_cart
        self.cb_pts_3D[:2, :] = self.cb_pts_2D_cart
        self.cb_pts_2D[2, :] = 1
        self.cb_pts_3D[3, :] = 1

    def getIndex(self, row, col):
        return row * self.cols + col

    def draw(self, rt_list):
        for rt in rt_list:
            # print("self.cb_pts_3D:  ", self.cb_pts_3D)
            rt_tensor = torch.tensor(rt, dtype=torch.float)
            hom_points = rt_tensor @ self.cb_pts_3D
            cart_points = hom_points[:-1] / hom_points[-1]
            # print("cart_points: ", cart_points)

            glBegin(GL_POINTS)
            for i in range(cart_points.size(1)):  # Loop through columns
                x, y, z = cart_points[:, i].tolist()
                glVertex3f(x, y, z)
            glEnd()

            glLineWidth(2)  # Set line width
            glBegin(GL_LINES)
            for i in range(0, self.lines.size(1)):
                p0 = int(self.lines[0,i].item())
                p1 = int(self.lines[1,i].item())
                x0, y0, z0 = cart_points[:, p0].tolist()
                x1, y1, z1 = cart_points[:, p1].tolist()
                glVertex3f(x0, y0, z0)
                glVertex3f(x1, y1, z1)
            glEnd()
