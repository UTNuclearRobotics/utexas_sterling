import cv2
import numpy as np

from camera_intrinsics import CameraIntrinsics
from homography_utils import *
from utils import *
from scipy.optimize import minimize, differential_evolution
import tkinter as tk


class HomographyFromChessboardImage:
    def __init__(self, image, cb_rows, cb_cols):
        # super().__init__(torch.eye(3))
        self.image = image
        self.cb_rows = cb_rows
        self.cb_cols = cb_cols
        self.chessboard_size = (cb_rows, cb_cols)

        # Get image chessboard corners, cartesian NX2
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (cb_cols, cb_rows), None)
        self.corners = corners.reshape(-1, 2)
        self.cb_tile_width = int(self.chessboard_tile_width())

        # Get model chessboard corners, cartesian NX2
        model_chessboard_2d = compute_model_chessboard_2d(cb_rows, cb_cols, self.cb_tile_width, center_at_zero=True)

        self.H, mask = cv2.findHomography(model_chessboard_2d, self.corners, cv2.RANSAC)
        self.K, K_inv = CameraIntrinsics().get_camera_calibration_matrix()
        self.RT, self.plane_normal, self.plane_distance = decompose_homography(self.H, self.K)

        self.validate_chessboard_2d(model_chessboard_2d)

        # Transform model chessboard 3D points to image points
        model_chessboard_3d = compute_model_chessboard_3d(cb_rows, cb_cols, self.cb_tile_width, center_at_zero=True)
        self.validate_chessboard_3d(model_chessboard_3d)

    def validate_chessboard_2d(self, model_chessboard_2d):
        # Transform model chessboard 2D points to image points
        self.transformed_model_chessboard_2d = self.transform_points(model_chessboard_2d.T, self.H)
        self.transformed_model_chessboard_2d = self.transformed_model_chessboard_2d.T.reshape(-1, 2).astype(np.float32)

    def validate_chessboard_3d(self, model_chessboard_3d):
        RT = self.get_rigid_transform()
        K = self.get_camera_intrinsics()
        self.IEK = K @ RT[:3] @ model_chessboard_3d.T
        self.model_cb_3d_to_2d = hom_to_cart(self.IEK)
        return self.model_cb_3d_to_2d

    def get_rigid_transform(self):
        return self.RT

    def get_camera_intrinsics(self):
        return self.K

    def get_plane_norm(self):
        return self.plane_normal

    def get_plane_dist(self):
        return self.plane_distance

    def chessboard_tile_width(self):
        """Calculate the maximum distance between two consecutive corners in each row of the chessboard."""
        # Sort corners by y value to group them by rows
        sorted_corners = sorted(self.corners, key=lambda x: x[1])

        # Split sorted_corners into rows
        interval = self.cb_cols
        rows = [sorted_corners[i * interval : (i + 1) * interval] for i in range(len(sorted_corners) // interval)]

        # Calculate distances between consecutive points in each row
        cb_tile_width = 0
        for row in rows:
            row.sort(key=lambda x: x[0])
            for i in range(len(row) - 1):
                distance = np.linalg.norm(np.array(row[i]) - np.array(row[i + 1]))
                cb_tile_width = max(cb_tile_width, distance)

        return cb_tile_width

    def validate_homography(self):
        keepRunning = True
        counter = 0
        cv2.namedWindow("Chessboard")
        while keepRunning:
            match counter % 3:
                case 0:
                    rend_image = draw_points(self.image, self.corners, color=(255, 0, 0))
                    cv2.setWindowTitle("Chessboard", "Original corners")
                case 1:
                    rend_image = draw_points(self.image, self.transformed_model_chessboard_2d, color=(0, 255, 0))
                    cv2.setWindowTitle("Chessboard", "Transformed 2D model chessboard corners")
                case 2:
                    rend_image = draw_points(self.image, self.model_cb_3d_to_2d.T, color=(0, 0, 255))
                    cv2.setWindowTitle("Chessboard", "Transformed 3D model chessboard corners")
            counter += 1
            cv2.imshow("Chessboard", rend_image)
            key = cv2.waitKey(0)
            if key == 113:  # Hitting 'q' quits the program
                keepRunning = False
        exit(0)

    def transform_points(self, points, H):
        """Transform points using the homography matrix."""
        hom_points = cart_to_hom(points)
        transformed_points = H @ hom_points
        return hom_to_cart(transformed_points)

    def plot_BEV_chessboard(self):
        image = self.image.copy()
        model_chessboard_2d_centered = compute_model_chessboard_2d(
            self.cb_rows, self.cb_cols, self.cb_tile_width, center_at_zero=False
        )
        H, mask = cv2.findHomography(model_chessboard_2d_centered, self.corners, cv2.RANSAC)
        dimensions = (int(self.cb_tile_width * (self.cb_cols - 1)), int(self.cb_tile_width * (self.cb_rows - 1)))
        warped_image = cv2.warpPerspective(image, np.linalg.inv(H), dsize=dimensions)

        cv2.imshow("BEV", warped_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        exit(0)

    import numpy as np

    def quadrilateral_area(self, points: np.ndarray) -> float:
        """
        Computes the area of a quadrilateral from four vertices (A, B, C, D).

        Parameters
        ----------
        points : np.ndarray
            - Shape (4, 2) for 2D points: [[Ax, Ay], [Bx, By], [Cx, Cy], [Dx, Dy]].
            - Shape (4, 3) for 3D points: [[Ax, Ay, Az], [Bx, By, Bz], ...].

        Returns
        -------
        float
            The area of the quadrilateral.

        Notes
        -----
        - For 2D, we assume the points are in a consistent order around the shape
        (e.g., A->B->C->D->A) and the quadrilateral is non-self-intersecting.
        - For 3D, we assume the four points are coplanar. We subdivide into
        triangles (A,B,C) and (A,C,D) and sum the areas.
        - If your quadrilateral is self-intersecting, or if 3D points are not
        coplanar, you may get incorrect or unexpected results.
        """

        # 2D case: Shoelace formula
        if points.shape == (4, 2):
            x = points[:, 0]
            y = points[:, 1]
            # Shoelace sums
            # area = 1/2 * abs( x0*y1 + x1*y2 + x2*y3 + x3*y0
            #                 - (y0*x1 + y1*x2 + y2*x3 + y3*x0 ) )
            area = 0.5 * abs(
                x[0] * y[1]
                + x[1] * y[2]
                + x[2] * y[3]
                + x[3] * y[0]
                - (y[0] * x[1] + y[1] * x[2] + y[2] * x[3] + y[3] * x[0])
            )
            return area

        # 3D case: Subdivide into two triangles, use cross products
        elif points.shape == (4, 3):
            A, B, C, D = points  # Unpack the rows for clarity
            # Triangle ABC area
            area_ABC = 0.5 * np.linalg.norm(np.cross(B - A, C - A))
            # Triangle ACD area
            area_ACD = 0.5 * np.linalg.norm(np.cross(C - A, D - A))
            return area_ABC + area_ACD

        else:
            raise ValueError("Input must have shape (4,2) or (4,3).")

    def submit(self):
        print("OK")
        self.theta = float(self.entries[0].get())
        self.x0 = float(self.entries[1].get())
        self.x1 = float(self.entries[2].get())
        self.y0 = float(self.entries[3].get())
        self.y1 = float(self.entries[4].get())
        for i, field in enumerate(self.fields):
            print(field, ": ", self.entries[i].get())

    def BEVEditor(self):
        root = tk.Tk()
        root.title("BEV Editor")
        self.fields = ["Theta", "x0", "x1", "y0", "y1"]
        self.entries = []
        for i, field in enumerate(self.fields):
            label = tk.Label(root, text=field)
            label.grid(row=i, column=0, padx=10, pady=5, sticky="e")
            entry = tk.Entry(root)
            entry.grid(row=i, column=1, padx=10, pady=5)
            self.entries.append(entry)
        submit_button = tk.Button(root, text="Submit", command=lambda: self.submit())
        submit_button.grid(row=len(self.fields), column=0, padx=10, pady=10)
        self.entries[0].delete(0, tk.END)
        self.entries[0].insert(0, "0.0")
        self.entries[1].delete(0, tk.END)
        self.entries[1].insert(0, "-10.0")
        self.entries[2].delete(0, tk.END)
        self.entries[2].insert(0, "10.0")
        self.entries[3].delete(0, tk.END)
        self.entries[3].insert(0, "-10.0")
        self.entries[4].delete(0, tk.END)
        self.entries[4].insert(0, "10.0")
        self.submit()

        while True:
            root.update_idletasks()  # Update "idle" tasks (like geometry management)
            root.update()

            RT = self.get_rigid_transform()
            print("RT:  ", RT)
            K = self.get_camera_intrinsics()
            model_rect_3d_hom = compute_model_rectangle_3d_hom(self.theta, self.x0, self.y0, self.x1, self.y1)
            model_rect_3d_applied_RT = K @ RT @ model_rect_3d_hom.T
            model_rect_2d = hom_to_cart(model_rect_3d_applied_RT)
            rend_image = draw_points(self.image, model_rect_2d.T, color=(255, 0, 255))
            cv2.imshow("Full BEV", rend_image)
            cv2.waitKey(1)


    def plot_BEV_full(self, plot_BEV_full=False):
        """
        Plots the bird's-eye view (BEV) image using optimized rectangle parameters.
        """

        RT = self.get_rigid_transform()
        K = self.get_camera_intrinsics()

        # Get optimized parameters
        theta, x1, y1, x2, y2 = optimize_rectangle_parameters(self.image, RT, K)

        # Generate optimized 3D rectangle
        model_rect_3d_hom = compute_model_rectangle_3d_hom(theta, x1, y1, x2, y2)
        model_rect_3d_applied_RT = K @ RT[:3] @ model_rect_3d_hom.T
        model_rect_2d = hom_to_cart(model_rect_3d_applied_RT)

        # Align rectangle with the bottom of the image
        model_rect_2d[1] -= model_rect_2d[1].max() - (self.image.shape[0] - 1)

        x_dif = abs(x2) + abs(x1)
        y_dif = abs(y2) + abs(y1)
        dsize = (int(x_dif), int(y_dif))

        # Adjust rectangle for warp perspective
        src_points = model_rect_2d.T[:, :2]

        # Define destination points aligned with the bottom
        dst_points = np.array([[0, 0], [dsize[0], 0], [dsize[0], dsize[1]], [0, dsize[1]]], dtype=np.float32)

        H, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC)
        warped_image = cv2.warpPerspective(self.image, H, dsize)

        # Resize the warped image
        warped_image = cv2.resize(warped_image, (int(dsize[0] / 5), int(dsize[1] / 5)))

        if plot_BEV_full:
            plot_BEV(self.image, model_rect_2d, warped_image)

        return H, dsize
