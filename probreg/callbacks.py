import copy
import open3d as o3
import matplotlib.pyplot as plt

try:
    import cupy as cp
    asnumpy = cp.asnumpy
except:
    def asnumpy(x):
        return x


class Plot2DCallback(object):
    """Display the 2D registration result of each iteration.

    Args:
        source (numpy.ndarray): Source point cloud data.
        target (numpy.ndarray): Target point cloud data.
        save (bool, optional): If this flag is True,
            each iteration image is saved in a sequential number.
    """
    def __init__(self, source, target, save=False,
                 keep_window=True):
        self._source = source
        self._target = target
        self._result = copy.deepcopy(self._source)
        self._save = save
        self._cnt = 0
        plt.axis('equal')
        source = asnumpy(self._source)
        target = asnumpy(self._target)
        result = asnumpy(self._result)
        plt.plot(source[:, 0], source[:, 1], 'ro', label='source')
        plt.plot(target[:, 0], target[:, 1], 'g^', label='target')
        plt.plot(result[:, 0], result[:, 1], 'bo', label='result')
        plt.legend()
        plt.draw()

    def __call__(self, transformation):
        self._result = transformation.transform(self._source)
        plt.cla()
        plt.axis('equal')
        source = asnumpy(self._source)
        target = asnumpy(self._target)
        result = asnumpy(self._result)
        plt.plot(source[:, 0], source[:, 1], 'ro', label='source')
        plt.plot(target[:, 0], target[:, 1], 'g^', label='target')
        plt.plot(result[:, 0], result[:, 1], 'bo', label='result')
        plt.legend()
        if self._save:
            plt.savefig('image_%04d.png' % self._cnt)
        plt.draw()
        plt.pause(0.001)
        self._cnt += 1


class Open3dVisualizerCallback(object):
    """Display the 3D registration result of each iteration.

    Args:
        source (numpy.ndarray): Source point cloud data.
        target (numpy.ndarray): Target point cloud data.
        save (bool, optional): If this flag is True,
            each iteration image is saved in a sequential number.
        keep_window (bool, optional): If this flag is True,
            the drawing window blocks after registration is finished.
        fov: Field of view (degree).
    """
    def __init__(self, source, target, save=False,
                 keep_window=True, fov=None):
        self._vis = o3.visualization.Visualizer()
        self._vis.create_window(width=1024, height=768)
        self._source = source
        self._target = target
        self._result = copy.deepcopy(self._source)
        self._save = save
        self._keep_window = keep_window
        self._source.paint_uniform_color([0.9, 0.1, 0.1])
        self._target.paint_uniform_color([0.1, 0.9, 0.1])
        self._result.paint_uniform_color([0.1, 0.1, 0.9])
        self._vis.get_render_option().point_size = 1
        self._vis.add_geometry(self._source)
        self._vis.add_geometry(self._target)
        self._vis.add_geometry(self._result)
        if not fov is None:
            ctr = self._vis.get_view_control()
            ctr.change_field_of_view(step=fov)
            ctr.set_front([ 0.71611399214239602, -0.55250678197254455, 0.4265172987490935 ])
            ctr.set_lookat([ 0.89838318938412876, 1.9992217088956057, 1.4572879726068768 ])
            ctr.set_up([ -0.3129388391381277, 0.29206210644233466, 0.903752736615136 ])
            ctr.set_zoom(0.71999999999999997)
        self._cnt = 0

    def __del__(self):
        if self._keep_window:
            self._vis.run()
        self._vis.destroy_window()

    def __call__(self, transformation):
        self._result.points = transformation.transform(self._source.points)
        self._vis.update_geometry(self._source)
        self._vis.update_geometry(self._target)
        self._vis.update_geometry(self._result)
        self._vis.poll_events()
        self._vis.update_renderer()
        if self._save:
            self._vis.capture_screen_image("image_%04d.jpg" % self._cnt,do_render=True)
        self._cnt += 1
