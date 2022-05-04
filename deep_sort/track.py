# vim: expandtab:ts=4:sw=4



class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.
    #单个目标跟踪状态的枚举类型。新建的跟踪是被归类为“暂定”直到收集到足够的证据。
    #然后，轨道状态更改为“确认”。不再活着的轨迹被分类为'删除'，以标记他们从一组活跃删除。
    """
    # 三种状态   尝试性的，确定的，被删除的
    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    # 具有状态空间的单个目标轨道（x，y，A，H）和相关联速度，其中“（x，y）”是bbox的中心，A是高宽比和“H”是高度。
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        # 初始状态分布的平均向量。
        Mean vector of the initial state distribution.
    covariance : ndarray #协方差
        # 初始状态分布的协方差矩阵
        Covariance matrix of the initial state distribution.
    track_id : int
        # 唯一的轨迹ID
        A unique track identifier.
    n_init : int
        # 在轨道设置为confirmed之前的连续检测帧数。当一个miss发生时，轨道状态设置为Deleted帧。
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        # 在侦测状态设置成Deleted前，最大的连续miss数。
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
         # 特征向量检测的这条轨道的起源。如果为空，则这个特性被添加到'特性'缓存中。
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray #均值
        # 初始分布均值向量。
        Mean vector of the initial state distribution.
    covariance : ndarray #协方差
        # 初始分布协方差矩阵。
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
         #测量更新的总数。
        Total number of measurement updates.
     hit_streak : int
        # 自上次miss之后，连续测量更新的总数。（更新一次+1）
        Total number of consective measurement updates since last miss.
    age : int
        #从开始的总帧数
        Total number of frames since first occurance.
    time_since_update : int
        # 从上次的测量更新完后，统计的总帧数
        Total number of frames since last measurement update.
    state : TrackState
         # 当前的侦测状态
        The current track state.
    features : List[ndarray]
        # 特性的缓存。在每个度量更新中，相关的特性向量添加到这个列表中。
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """
    #初始化各参数
    def __init__(self, mean, covariance, track_id, n_init, max_age,
                 feature=None):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = TrackState.Tentative #初始为待定状态
        self.features = []
        if feature is not None:
            self.features.append(feature) #特征入库

        self._n_init = n_init
        self._max_age = max_age

    # 将bbox转换成xywh
    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    # 获取当前位置以某种格式，以边界框的格式(min x ,min y,max x ,max y)表示当前位置,返回值为边界框
    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    # 预测，基于kalman filter
    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    # 更新。 主要是步进和特征，检测方法为级联检测
    def update(self, kf, detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        self.features.append(detection.feature)

        self.hits += 1
        self.time_since_update = 0 #重置为0
       # 满足条件时确认追踪器
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    # 标记已经miss的，如果从更新起miss了_max_age（30）帧以上，设置为Deleted
    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        # 待定状态的追踪器直接删除
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        # 已经时confirm状态的追踪器，虽然连续多帧对目标进行了预测，
        # 但中间过程中没有任何一帧能够实现与检测结果的关联，说明目标
        # 可能已经移除了画面，此时直接设置追踪器为待删除状态
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    # 设置三种状态
    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted
