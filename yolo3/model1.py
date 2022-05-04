"""YOLO_v3 Model Defined in Keras."""

from functools import wraps

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2

from yolo3.utils import compose


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """
    Wrapper to set Darknet parameters for Convolution2D.
    Model是Keras中最主要的数据结构之一，该数据结构定义了一个完整的图。
    因此，所有的描述都是在讲解darknet53这个神经网络的结构和数据流动的路径，而不是在具体执行一个函数或一段程序，
    这样就比较容易理解这里的每一个DarknetConv2d()函数的输入为什么只有输入张量，而不需要每次匹配对应的卷积核
    （因为训练好的卷积核会通过load_weights方法加载到Model中，并与每一个卷积计算相匹配）。
    """
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}   #将核权重矩阵的正则化，使用L2正则化，参数是5e-4，将该核权重参数W进行正则化#######################question：权重矩阵正则化？
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'  #Padding，一般使用same模式，只有当步长为(2,2)时，使用valid模式。避免在降采样中，引入无用的边界信息；
    darknet_conv_kwargs.update(kwargs)   #更新，其余参数不变，都与二维卷积操作Conv2D()一致。
    return Conv2D(*args, **darknet_conv_kwargs)  #所谓2D卷积，就是横向和纵向均卷积

def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),    #################################################################question：BatchNormalization（）？
        LeakyReLU(alpha=0.1))

def resblock_body(x, num_filters, num_blocks):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1,0),(1,0)))(x)   #填充x的边界为0，由(?, 416, 416, 32)转换为(?, 417, 417, 32)。因为下一步卷积操作的步长为2，所以图的边长需要是奇数；
    x = DarknetConv2D_BN_Leaky(num_filters, (3,3), strides=(2,2))(x)      #DarkNet的2维卷积操作，核是(3,3)，步长是(2,2)，因此会导致特征尺寸变小，由(?, 417, 417, 32)转换为(?, 208, 208, 64)。由于num_filters是64，所以产生64个通道。
    for i in range(num_blocks):
        y = compose(             #compose：输出预测图y，功能是组合函数，先执行1x1的卷积操作，再执行3x3的卷积操作，过滤器数量先减半，再恢复，最后与输入相同，都是64；
                DarknetConv2D_BN_Leaky(num_filters//2, (1,1)), #########################################################question:步长为1？卷积核数减半？
                DarknetConv2D_BN_Leaky(num_filters, (3,3)))(x)
        x = Add()([x,y])           #残差操作，将x的值与y的值相加。残差操作可以避免在网络较深时所产生的梯度弥散问题。
    return x

def darknet_body(x):
    '''Darknent body having 52 Convolution2D layers'''
    x = DarknetConv2D_BN_Leaky(32, (3,3))(x)
    x = resblock_body(x, num_filters=64, num_blocks=1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)
    return x

def make_last_layers(x, num_filters, out_filters):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    x = compose(
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)))(x)
    y = compose(
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D(out_filters, (1,1)))(x)   #不含BN和Leaky的卷积操作——>相当于全连接操作
    return x, y


def yolo_body(inputs, num_anchors, num_classes):
    """Create YOLO_V3 model CNN body in Keras."""
    darknet = Model(inputs, darknet_body(inputs))
    x ,y1= compose(
            DarknetConv2D_BN_Leaky(256, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,darknet.layers[152].output])   #darknet.layers[152].output.shape = [?,26,26,512]   #152层 = 3+（4+7）+（4+7*2）+（4+7*8）+（4+7*8）-->darkent53:43
    x, y2 = make_last_layers(x, 256, num_anchors*(num_classes+5)) #x.shape = [?,26,26,256],y2.shape = [?,26,26,num_anchors*(num_classes+5)]



    x = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,darknet.layers[92].output])   #darknet.layers[92].output.shape = [?,52,52,256]
    x, y3 = make_last_layers(x, 128, num_anchors*(num_classes+5))

    return Model(inputs, [y1,y2,y3])


def yolo_head(feats, anchors, num_classes, input_shape):
    # feats，即：feature maps
    """Convert final layer features to bounding box parameters."""
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    grid_shape = K.shape(feats)[1:3] # height, width #13x13或26x26或52x52
    # 通过arange、reshape、tile的组合，根据grid_shape(13x13、26x26或52x52）创建y轴的0~N-1的组合grid_y，再创建x轴的0~N-1的组合grid_x，将两者拼接concatenate，形成NxN的grid(13x13、26x26或52x52）
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))   ################################################################question-3:grid  shape->[feats[0],feats[1],1,1]

    # 从待处理的feature map的最后一维数据中，先将num_anchors这个维度与num_classes+5这个维度的数据分离，再取出4个框值tx、ty（最后一维数据的0:1）、tw和th（最后一维数据的2:3）、置信度（最后一维数据4）
    feats = K.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5]) ########################################################question-4：grid_shape[0], grid_shape[1]
    # Adjust preditions to each spatial grid point and anchor size.

    # 用sigmoid()函数计算目标框的中心点box_xy，用exp()函数计算目标框的宽和高box_wh
    # 使用特征图尺寸（如：13x13、26x26或52x52）在水平x、垂直y两个维度对box_xy进行归一化，确定目标框的中心点的相对位置
    # 使用标准图片尺寸（416x416）在宽和高两个维度对box_wh（因为，3组9个anchor box是基于416x416尺寸定义的）进行归一化，确定目标框的高和宽的相对位置
    box_xy = K.sigmoid(feats[..., :2])
    box_wh = K.exp(feats[..., 2:4])
    #用sigmoid()函数计算目标框的置信度box_confidence
    box_confidence = K.sigmoid(feats[..., 4:5])
    # 用sigmoid()函数计算目标框的类别置信度box_class_probs
    box_class_probs = K.sigmoid(feats[..., 5:])

    # Adjust preditions to each spatial grid point and anchor size.
    box_xy = (box_xy + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = box_wh * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))

    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    '''
    Get corrected boxes
    对模型输出的box信息(x, y, w, h)进行校正,输出基于原图坐标系的box信息(x_min, y_min, x_max, y_max)
    Args:
        box_xy: 模型输出的box中心坐标信息,范围0~1
        box_wh: 模型输出的box长宽信息,范围0~1
        input_shape: 模型的图像尺寸, 长宽均是32倍数
        image_shape: 原图尺寸
    Returns:
        boxes: 基于原图坐标系的box信息(实际坐标值,非比值)

    '''
    # 将box_xy, box_wh转换为输入图片上的真实坐标，输出boxes是框的左下、右上两个坐标(y_min, x_min, y_max, x_max)
    # np.array[i:j:s]，当s<0时，i缺省时，默认为-1；j缺省时，默认为-len(a)-1；所以array[::-1]相当于array[-1:-len(a)-1:-1]，也就是从最后一个元素到第一个元素复制一遍，即倒序
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape/image_shape))    #################################################################question-5:调整的目的
    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    '''
    Process Conv layer output
    :Input:
        # feats：需要处理的featue map
        # shape：(?,13,13,255)，(?,26,26,255)或(?,52,52,255)
        # anchors：每层对应的3个anchor box
        # num_classes：类别数（80）
        # input_shape:（416,416）
        # image_shape：图像尺寸
    :returns
        boxes:
        box_scores:
    '''

    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats,
        anchors, num_classes, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])     #形成框的列表boxes（?, 4)
    box_scores = box_confidence * box_class_probs         #框的得分=框的置信度x类别置信度
    box_scores = K.reshape(box_scores, [-1, num_classes])      #形成框的得分列表box_scores（?, 80)
    return boxes, box_scores


def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=20,          #每张图每类最多检测到20个框
              score_threshold=.6,
              iou_threshold=.5):
    """Evaluate YOLO model on given input and return filtered boxes."""
    # 将anchor_box分为3组，分别分配给13x13、26x26、52x52等3个yolo_model输出的feature map
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]                  ################################################################question-1:anchor_mask
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32         ##################################################################question-2:yolo_outputs
    boxes = []
    box_scores = []
    # 分别对3个feature map运行
    for l in range(3):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
            anchors[anchor_mask[l]], num_classes, input_shape, image_shape)     ##################################################question-3：anchor_mask[]
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    # 将运算得到的目标框用concatenate()函数拼接为（?, 4)的元组，将目标框的置信度拼接为(?,1)的元组
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    # 计算MASK掩码，过滤小于score阈值的值，只保留大于阈值的值
    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        # TODO: use keras backend instead of tf.
        class_boxes = tf.boolean_mask(boxes, mask[:, c])         #通过掩码MASK和类别C筛选框boxes
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])           #通过掩码MASK和类别C筛选scores
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)  #运行非极大抑制non_max_suppression（），每类最多检测20个框
        # K.gather:根据索引nms_index选择class_boxes和class_box_scores，标出选出的框的类别classes   ##################################################question：Mask和non_max_suppression是否冗余
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c    ######################################################question：classes矩阵中乘以c？
        # 用concatenate()函数把选出的class_boxes、class_box_scores和classes拼接，形成(?,4)、(?,1)和(?,80)的元组返回
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    '''Preprocess true boxes to training input format

    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5)
        Absolute x_min, y_min, x_max, y_max, class_code reletive to input_shape.
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), wh
    num_classes: integer

    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value

    '''
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy/input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh/input_shape[::-1]

    m = true_boxes.shape[0]
    grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(3)]
    y_true = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+num_classes),
        dtype='float32') for l in range(3)]

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0]>0

    for b in range(m):
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(3):
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[b,t,0]*grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b,t,1]*grid_shapes[l][0]).astype('int32')
                    n = anchor_mask[l].index(n)
                    c = true_boxes[b,t, 4].astype('int32')
                    y_true[l][b, j, i, n, 0:4] = true_boxes[b,t, 0:4]
                    y_true[l][b, j, i, n, 4] = 1
                    y_true[l][b, j, i, n, 5+c] = 1
                    break

    return y_true

def box_iou(b1, b2):
    '''Return iou tensor

    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)

    '''

    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou



def yolo_loss(args, anchors, num_classes, ignore_thresh=.5):
    '''Return yolo_loss tensor

    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(T, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss

    Returns
    -------
    loss: tensor, shape=(1,)

    '''
    yolo_outputs = args[:3]
    y_true = args[3:]
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0])) #13*32=416  input_shape--->[416,416]
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(3)] #(13,13),(26,26),(52,52)
    loss = 0
    m = K.shape(yolo_outputs[0])[0] # batch size, tensor
    # 逐层计算损失
    for l in range(3):
        object_mask = y_true[l][..., 4:5] # 取出置信度
        true_class_probs = y_true[l][..., 5:] #取出类别信息
        # yolo_head讲预测的偏移量转化为真实值，这里的真实值是用来计算iou,并不是来计算loss的，loss使用偏差来计算的
        pred_xy, pred_wh, pred_confidence, pred_class_probs = yolo_head(yolo_outputs[l],
             anchors[anchor_mask[l]], num_classes, input_shape)  #anchor_mask[0]=[6,7,8]
        pred_box = K.concatenate([pred_xy, pred_wh])  #anchors[anchor_mask[l]]=array([[  116.,   90.], [  156., 198.],[  373., 326.]])

        # Darknet raw box to calculate loss.
        xy_delta = (y_true[l][..., :2]-pred_xy)*grid_shapes[l][::-1] #根据公式将boxes中心点x,y的真实值转换为偏移量
        wh_delta = K.log(y_true[l][..., 2:4]) - K.log(pred_wh) #计算宽高的偏移量
        # Avoid log(0)=-inf.
        wh_delta = K.switch(object_mask, wh_delta, K.zeros_like(wh_delta))
        box_delta = K.concatenate([xy_delta, wh_delta], axis=-1)
        box_delta_scale = 2 - y_true[l][...,2:3]*y_true[l][...,3:4] #( 2-box_ares）避免大框的误差对loss 比小框误差对loss影响大

        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True) #定义一个size可变的张量来存储不含有目标的预测框的信息
        object_mask_bool = K.cast(object_mask, 'bool') #映射成bool类型  1=true 0=false
        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[l][b,...,0:4], object_mask_bool[b,...,0]) #剔除为0的行
            iou = box_iou(pred_box[b], true_box)   #一张图片预测出的所有boxes与所有的ground truth boxes计算iou 计算过程与生成label类似利用了广播特性这里不详细描述
            best_iou = K.max(iou, axis=-1)  #找出最大iou
            ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box))) #当iou小于阈值时记录，即认为这个预测框不包含物体
            return b+1, ignore_mask
        _, ignore_mask = K.control_flow_ops.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask]) #传入loop_body函数初值为b=0，ignore_mask
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)  #扩展维度用来后续计算loss

        box_loss = object_mask * K.square(box_delta*box_delta_scale)
        # 置信度损失既包含有物体的损失 也包含无物体的置信度损失
        confidence_loss = object_mask * K.square(1-pred_confidence) + \
            (1-object_mask) * K.square(0-pred_confidence) * ignore_mask
        # 分类损失只计算包含物体的损失
        class_loss = object_mask * K.square(true_class_probs-pred_class_probs)
        loss += K.sum(box_loss) + K.sum(confidence_loss) + K.sum(class_loss)
    return loss / K.cast(m, K.dtype(loss))
