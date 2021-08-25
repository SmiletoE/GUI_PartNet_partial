'''
负责模型调用，主要算法
'''
import os
import sys
import tensorflow as tf
import numpy as np

import heapq
GPU_INDEX = 0
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, './model/ins_seg_detection')) #动态找地址

path = './model/ins_seg_detection'
import model
def pre(input,algorithm,catgory,level):
    input = np.repeat(input[np.newaxis, ...], 32, axis=0) #为了调用模型，又改不了batch_size，所以复制了32遍匹配原输入维度，但这儿还有个问题，作者在训练时有的又是用的24的size,所以还需对不同种类硬编码才能调用所有模型
    input = tf.convert_to_tensor(input,dtype='float32')

    epoch = '249'
    if catgory in (['Refrigerator']) and level == '1':
        epoch = '149'
    elif catgory in (['StorageFurniture']):
        if level == '1':
            epoch = '149'
        elif level == '2':
            epoch = '150'
        elif level == '3':
            epoch = '129'
    elif catgory in (['TrashCan']) and level == '1':
        epoch = '089'
    elif catgory in (['Bag','Bed','Bottle','Bowl','Clock','Dishwasher','Display','Door','Earphone','Faucet','Hat','Keyboard','Knife','Laptop','Microwave','Mug','Refrigerator','Scissors','Vase']):
        epoch = '249'
    elif catgory in (['Chair','Lamp','Table',]):
        epoch = '149'

    with tf.Session() as sess:
        with tf.device('/gpu:' + str(GPU_INDEX)):
            config = tf.ConfigProto()  # tf.ConfigProto()主要的作用是配置tf.Session的运算方式，比如gpu运算或者cpu运算
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False
            sess = tf.Session(config=config)

            # 加载元图和权重
            saver = tf.train.import_meta_graph(path + '/%s-%s/trained_models/epoch-%s.ckpt.meta'%(catgory,level,epoch))
            saver.restore(sess, tf.train.latest_checkpoint(path + "/%s-%s/trained_models/"%(catgory,level)))

            graph = tf.get_default_graph()  #获取当前默认计算图


            # layers = [op.name for op in graph.get_operations()] #获取所有操作
            # print(graph.as_graph_def().node)
            # print(layers)
            # pc_pl = graph.get_tensor_by_name("pc_pl:0")
            # print(pc_pl)
            input = sess.run(input)
            Placeholder = graph.get_tensor_by_name('Placeholder:0')
            print(Placeholder)
            Placeholder_5 = graph.get_operation_by_name('Placeholder_5').outputs[0]
            seg_net = graph.get_operation_by_name('seg/fc4/BiasAdd').outputs[0]
            ins_net = graph.get_operation_by_name('Reshape_1').outputs[0]
            mask_pred = graph.get_operation_by_name('transpose/perm').outputs[0]
            other_mask_pred = graph.get_operation_by_name('transpose').outputs[0]
            conf_net = graph.get_operation_by_name('Sigmoid').outputs[0]
            feed_dict = {Placeholder:input,
                         Placeholder_5.name:False}
            seg_net = sess.run(seg_net, feed_dict)[0]
            ins_net = sess.run(ins_net, feed_dict)[0]
            # mask_pred = sess.run(mask_pred, feed_dict)
            # other_mask_pred = sess.run(other_mask_pred, feed_dict)[0]
            conf_net =  sess.run(conf_net,feed_dict)[0]

            np.set_printoptions(threshold=np.inf)
            # print(seg_net.shape,ins_net.shape,mask_pred.shape,other_mask_pred.shape,conf_net.shape)
            # print(np.argmax(ins_net,axis=1))
            # print(mask_pred)
            # print(conf_net)
            # max_indexs = heapq.nlargest(6, range(len(conf_net)), conf_net.take) #获取最大N个下标
            # print(max_indexs)
            # print(np.argmax(conf_net))
            # print(np.argmax(seg_net,axis=1,))
            part_num = seg_net.shape[1]
            seg_net = np.argmax(seg_net,axis=1)
            ins_net = np.argmax(ins_net,axis=1)
            max_indexs = heapq.nlargest(part_num, range(len(conf_net)), conf_net.take)  # 获取最大N个下标

            # GUI.frame.show_out_pts(point,seg_net)
    tf.reset_default_graph()
    return seg_net,ins_net,max_indexs







    #获取所有变量
    # from tensorflow.python import pywrap_tensorflow
    # checkpoint_path = os.path.join(path, "trained_models/epoch-249.ckpt")
    # reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    # var_to_shape_map = reader.get_variable_to_shape_map()
    # for key in var_to_shape_map:
    #     print("tensor_name: ", key)



        # model_reader = pywrap_tensorflow.NewCheckpointReader(path + "trained_models/epoch-249.ckpt")
        #
        # # 然后，使reader变换成类似于dict形式的数据
        # var_dict = model_reader.get_variable_to_shape_map()
        #
        # # 最后，循环打印输出
        # for key in var_dict:
        #     print("variable name: ", key)
        #     # print(model_reader.get_tensor(key))