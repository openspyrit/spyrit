# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 18:04:11 2023

@author: ducros
"""

#%% 
import torch
import numpy as np
from test_helpers import assert_test

def test_core_meas():
        #%% Test Linear
        from spyrit.core.meas import Linear

        # constructor
        H = np.array(np.random.random([400,1000]))
        meas_op = Linear(H)

        # forward
        x = torch.rand([10,1000], dtype=torch.float)
        y = meas_op(x)
        print('forward:', y.shape)
        assert_test(y.shape, torch.Size([10, 400]), "Wrong forward size")

        # adjoint
        x = torch.rand([10,400], dtype=torch.float)
        y = meas_op.adjoint(x)
        print('adjoint:', y.shape)
        assert_test(y.shape, torch.Size([10, 1000]), "Wrong adjoint size")

        # get_mat
        H = meas_op.get_H()
        print('get_mat:', H.shape)
        assert_test(H.shape, torch.Size([400, 1000]), "Wrong get_mat size")

        #%% Test LinearSplit
        from spyrit.core.meas import LinearSplit

        # constructor
        H = np.array(np.random.random([400,1000]))
        meas_op = LinearSplit(H)

        # forward
        x = torch.rand([10,1000], dtype=torch.float)
        y = meas_op(x)
        print('Forward:', y.shape)
        assert_test(y.shape, torch.Size([10, 800]), "Wrong forward size")

        # forward_H
        x = torch.rand([10,1000], dtype=torch.float)
        y = meas_op.forward_H(x)
        print('Forward_H:', y.shape)
        assert_test(y.shape, torch.Size([10, 400]), "Wrong forward_H size")

        # adjoint
        x = torch.rand([10,400], dtype=torch.float)
        y = meas_op.adjoint(x)
        print('Adjoint:', y.shape)
        assert_test(y.shape, torch.Size([10, 1000]), "Wrong adjoint size")

        # get_mat
        H = meas_op.get_H()
        print('Measurement matrix:', H.shape)
        assert_test(H.shape, torch.Size([400, 1000]), "Wrong measurement matrix size")

        #%% Test HadamSplit
        from spyrit.core.meas import HadamSplit

        # constructor
        Ord = np.random.random([32,32])
        meas_op = HadamSplit(400, 32, Ord)

        # forward
        x = torch.rand([10,32*32], dtype=torch.float)
        y = meas_op(x)
        print('Forward:', y.shape)
        assert_test(y.shape, torch.Size([10, 800]), "Wrong forward size")

        # forward_H
        x = torch.rand([10,32*32], dtype=torch.float)
        y = meas_op.forward_H(x)
        print('Forward_H:', y.shape)
        assert_test(y.shape, torch.Size([10, 400]), "Wrong forward_H size")

        # adjoint
        x = torch.rand([10,400], dtype=torch.float)
        y = meas_op.adjoint(x)
        print('Adjoint:', y.shape)
        assert_test(y.shape, torch.Size([10, 1024]), "Wrong adjoint size")

        # get_mat
        H = meas_op.get_H()
        print('Measurement matrix:', H.shape)
        assert_test(H.shape, torch.Size([400, 1024]), "Wrong measurement matrix size")

        # pinv
        y = torch.rand([85,400], dtype=torch.float)
        x = meas_op.pinv(y)
        print('Pinv:', x.shape)
        assert_test(x.shape, torch.Size([85, 1024]), "Wrong pinv size")

        # inverse
        y = torch.rand([85,32*32], dtype=torch.float)
        x = meas_op.inverse(y)
        print('Inverse:', x.shape)
        assert_test(x.shape, torch.Size([85, 1024]), "Wrong inverse size")

        #%% Test LinearRowSplit
        from spyrit.core.meas import LinearRowSplit

        # constructor
        H_pos = np.random.rand(24,64)
        H_neg = np.random.rand(24,64)
        meas_op = LinearRowSplit(H_pos,H_neg)

        # forward
        x = torch.rand([10,64,92], dtype=torch.float)
        y = meas_op(x)
        print(y.shape)
        assert_test(y.shape, torch.Size([10, 48, 92]), "Wrong forward size")

        # forward_H
        x = torch.rand([10,64,92], dtype=torch.float)
        y = meas_op(x)
        print(y.shape)
        assert_test(y.shape, torch.Size([10, 48, 92]), "Wrong forward size")

        # get_H
        H = meas_op.get_H()
        print(H.shape)
        assert_test(H.shape, torch.Size([23, 64]), "Wrong measurement matrix size")

        return(True)

if __name__ == '__main__':
    test_core_meas()

