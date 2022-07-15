import os
import time

import cv2
import numpy as np

import torch
try:
    from torch2trt import torch2trt
except ImportError:
    raise ImportError('Please ensure that you install torch2trt!')

import tensorrt as trt 

from torch.nn import  functional as F

from models.mbv2_mlsd_tiny import  MobileV2_MLSD_Tiny
from models.mbv2_mlsd_large import  MobileV2_MLSD_Large

from calibrator import ImageFolderCalibDataset

from argparse import ArgumentParser, SUPPRESS

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="model type",  choices=['tiny', 'large'], default='tiny')
    args.add_argument("-e", "--engine", help="converted engine path", type=str, default=None)
    args.add_argument("-c", "--conversion", help="Conversion type", choices=['fp16', 'int8', 'onnx'], default='fp16')
    args.add_argument("-cd", "--calibration_data", help="Path to int8 calibration data", type=str, default='')
    args.add_argument("-cb", "--calibration_batch", help="Calibration batch size", type=int, default=32)
    args.add_argument("-s", "--serialize", help="Serialize trt engine to disk", action="store_true")
    args.add_argument("-b", "--bench", help="Toggle simple inference cost analysis", action="store_true")

    return parser

def onnx_convert(dummy_input, model, model_path, opset=11, device='cpu'):
    print('converting to onnx...')

    out = f"{os.path.splitext(model_path)[0]}.onnx"
    
    model = MobileV2_MLSD_Tiny().to(device).eval()
    dummy_input = torch.randn(1, 4, 512, 512).float().to(device)
    model.load_state_dict(torch.load(model_path), strict=True)

    torch.onnx.export(model, 
                      dummy_input, 
                      out, 
                      verbose=True, 
                      opset_version=opset
                      )

    print(f'converted successfuly at: {out}')


def main(model_type='tiny', 
        conversion='fp16', 
        engine_path='', 
        serialize=False, 
        calibration_data='' ,
        calibration_batch=32, 
        bench=False
        ):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
    model_path = f'./models/mlsd_{model_type}_512_fp32.pth'
    
    model = MobileV2_MLSD_Tiny().cuda().eval()
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)


    dummy_input = torch.randn(1, 4, 512, 512).float().to(device)

    if conversion == 'fp16':

        out_path = f'./models/mlsd_{model_type}__512_trt_{conversion}.pth'


        model = torch2trt(model, 
                        [dummy_input], 
                        fp16_mode=True, 
                        log_level=trt.Logger.INFO, 
                        max_workspace_size= 1 << 28
                        )

        print(f'\nsaving model to {out_path}\n')
        torch.save(model.state_dict(), out_path)
    

    elif conversion == 'int8':

        out_path = f'./models/mlsd_{model_type}__512_trt_{conversion}.pth'

        assert os.path.exists(calibration_data) == True, 'Calibration path does not exist!'

        dataset = ImageFolderCalibDataset(calibration_data)
        model = torch2trt(model, 
                        [dummy_input], 
                        int8_mode=True,
                        fp16_mode=True,
                        int8_calib_dataset=dataset, 
                        int8_calib_batch_size=calibration_batch,
                        log_level=trt.Logger.INFO,
                        max_workspace_size= 1 << 28
                        )

        print(f'\nsaving model to {out_path}\n')
        torch.save(model.state_dict(), out_path)

    elif conversion == 'onnx':
        onnx_convert(dummy_input, model, model_path)


    if serialize and conversion != 'onnx':

        print('\nsaving serialized engine to:  {engine_path}\n')

        with open(engine_path, "wb") as f:
            f.write(model.engine.serialize())

    if bench and conversion != 'onnx':

        print('Benchmarking after warmup...\n')
    
        for i in range(500):
            output = model(dummy_input)

        torch.cuda.current_stream().synchronize()  
        t0 = time.monotonic()
        for i in range(100):
            output = model(dummy_input)
        
        it0 = time.monotonic()
        output = output = model(dummy_input)
        it1 = time.monotonic()
        
        torch.cuda.current_stream().synchronize() 
        t1 = time.monotonic()
        fps = 100.0 / (t1 - t0)

        print(f'FPS: {fps:.2f}')
        print(f'Inference cost: {(it1-it0)*1000:.2f} ms')

if __name__ == '__main__':
    
    args = build_argparser().parse_args()

    main(args.model,
        args.conversion,
        args.engine,
        args.serialize,
        args.calibration_data,
        args.calibration_batch,
        args.bench
        )

