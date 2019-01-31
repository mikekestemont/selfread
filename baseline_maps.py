""""
Compute map of baselines using a pre-trained ARU-Net model (pb file)

Remarks: some functionality is taken directly from the pixlab file (esp.
run_demo_inference.py)
"""

from __future__ import print_function, division

import argparse
import os
import numpy as np
from scipy import misc
import tensorflow as tf
from util import getFiles, getProgressBar, load_graph
from PIL import Image

# instead of skimage we could use opencv...
from skimage.morphology import skeletonize,binary_closing,binary_opening
from skimage.io import imread
from skimage.transform import rotate

import cv2 

class Inference_pb(object):
    """
        Perform inference for an arunet instance

        :param net: the arunet instance to train

    """
    def __init__(self, path_to_pb, files, scale=0.5, 
                 prune_method='simple',
                 outdir=None, out_suffix=None, gpu_device='0',
                 test_orientation=True):
        """
        parameter:
            path_to_pb: path to trained tensorflow pb file
            files: list of image file paths
            scale: scale-factor, typically ARU-Net works well on low scales,
                which speeds up the inference a lot
            prune_method: options=['simple'] TODO: extent
            out_dir: output folder
            out_suffix: suffix to append to the filename (also decideds file
            extension = png if nothing set)
            gpu_device: device number as string
            test_orientation: test for the orientation. Note: cannot test for
            flips / 180 degree orientation :/
        """
        self.graph = load_graph(path_to_pb)
        self.files = files
        self.scale = scale
        self.prune_method = prune_method
        self.outdir = outdir
        self.out_suffix = out_suffix
        self.gpu_device = gpu_device
        self.test_orientation = test_orientation

    def runSession(self, sess, x, predictor, small):
        """ 
        helper function
        """
        if small.ndim == 2:
            small = np.expand_dims(small,2)

        # currently a batch is a single image -> use more? 
        batch_x = np.expand_dims(small,0)
        # Run session
        pred = sess.run(predictor, feed_dict={x: batch_x})

        return pred

    def applyARUNet(self):
        """ 
        apply the ARU-Net to list of files and writes outputs to outdir
        """
        assert(self.files and len(self.files) > 0)

        # TODO: move this to the init method
        session_conf = tf.ConfigProto()
        session_conf.gpu_options.visible_device_list = self.gpu_device
        pred = None
        with tf.Session(graph=self.graph, config=session_conf) as sess:
            x = self.graph.get_tensor_by_name('inImg:0')
            predictor = self.graph.get_tensor_by_name('output:0')
            
            progress = getProgressBar()
            for i in progress(range(len(self.files))):
#                print(self.files[i])
                # img: numpy array (height x width x channels)
                # scipy's misc.imread is deprecated
                # TODO: switch maybe to opencv instead of pillow with its image
                # class overhead
                pil_img = Image.open(self.files[i]).convert('L') # grayscale
                img = np.array(pil_img)
                size = (int(img.shape[1]*self.scale),int(img.shape[0]*self.scale))
                small = np.array(pil_img.resize(size, resample=Image.BICUBIC))
                origsize = (img.shape[1],img.shape[0])   

                # TODO: can we actually put them in 1 or 2 batches?
                pred1 = self.runSession(sess, x, predictor, small)
                out = self.pruneBaselines(pred1, size=origsize)

                # try other orientation
                # TODO: think of a better check! 
                # this one depends on the resolution and due to noise might
                # still pass...
                if self.test_orientation and \
                    np.count_nonzero(out) < 100: 
                    print('WARNING: no baseline found for img:'
                          ' {}'.format(self.files[i]))
                    print('rotate it now and try again...')
                    # rotate 90 degree counter clock-wise
                    small2 = rotate(small, 90, True)
                    pred2 = self.runSession(sess, x, predictor, small2)                    
                    origsize = (origsize[1],origsize[0])
                    out2 = self.pruneBaselines(pred2, size=origsize)
                    # check which direction has higher probability
                    # Note: unfortunately the probas are similar high for 0 + 180,
                    # as well as 90 and 270 degree, so we cannot test for these
                    # orientations!
                    # Note 2: raw probability map didnt work out for me, so lets do
                    # it that way
                    n_comp, _, stats1, _ =\
                        cv2.connectedComponentsWithStats(out.astype(np.uint8))
                    n_comp2, _, stats2, _ =\
                        cv2.connectedComponentsWithStats(out2.astype(np.uint8))
                    # test for area, assumption is that we get larger
                    # mean/median/sum area if it's correctly rotated
                    # TODO: might still be a bad test due to noise...
                    stat1 = np.sum(stats1[1:,cv2.CC_STAT_AREA])
                    stat2 = np.sum(stats2[1:,cv2.CC_STAT_AREA])
                    if stat2 > stat1: 
                        print('rotation by 90 degree counter clockwise gives higher'
                              ' probability (orig {} vs rot: {}) for file {}\n'
                              ' -> rotate this file (90 degree'
                              ' counter clock-wise), too!'.format(stat1, stat2, self.files[i]))
                        out = out2
                
                # small check if we have found a line at all
                if np.count_nonzero(out) < 50: # TODO: think of a better check
                    print('WARNING: no baseline found for img:'
                          ' {}'.format(self.files[i]))

                # save it
                name = os.path.splitext(os.path.basename(self.files[i]))[0]
                suffix = self.out_suffix if self.out_suffix else ''
                path = os.path.join(self.outdir, '{}{}.png'.format(name,suffix))
#                print('save to: {}'.format(path))
                misc.imsave(path, out)
            
        return pred

    def pruneBaselines(self, aru_prediction, size=()):
        """
        computes from the aru_prediction the main baselines
        parameters:
            aru_prediction: prediction of the ARU-Net, i.e. 1 x height x width x
            chans, where chans[0] = baselines, chans[1] = begin/start, chans[2] =
            background
            size: original image size must be (width,height)
        returns:
            image encoded baselines (255: baseline point, 0: background)
        """
        if self.prune_method == 'simple':
            bl = aru_prediction[0,:,:,0] 
            other = aru_prediction[0,:,:,2] 
            # binarization
            b = 0.4 # see paper
            # take both classes into account
            out = np.where(np.logical_and(bl > b,other < b), 1.0, 0)
            # remove some holes and single items
            # important step, otherwise the skeleton will have many small
            # branches
            # TODO: exchange w. opencv counterpart (faster)
            selem = np.ones((1,3))
            out = np.where(binary_closing(out,selem=selem),1.0,0.0)
            out = np.where(binary_opening(out,selem=selem),1.0,0.0)
#            misc.imsave(os.path.join(self.outdir,'tmp.png'), out)

            # enlarge output again
            # out = misc.imresize(out, size, interp='nearest')            
            # deprecated, use:
            out = np.array(Image.fromarray(out).resize(size,
                                                       resample=Image.NEAREST))
            # now let's get only single pixel lines
#            misc.imsave(os.path.join(self.outdir,'tmp2.png'), out)
            out = skeletonize(out) 
            out = out * 255
        else:
            print('not implemented yet')

        return out


def parseArgs(parser):
    parser.add_argument('--file', 
                        help='single file')
    parser.add_argument('--indir',
                        help='the input folder of the images / features')
    parser.add_argument('--labels', 
                        help='contains images/descriptors to load + labels'
                       ' if not given search for files with given suffix')
    parser.add_argument('-s', '--suffix',
                        default='.jpg',
                        help='only chose those images with a specific suffix')
    parser.add_argument('--aru_model',
                        help='path to the ARU-Net model')
    parser.add_argument('--scale', type=float, default=0.25,
                        help='scale factor for processing, final line map will'
                        ' be upscaled again')
    parser.add_argument('-o', '--outdir',
                        help='path for outputfolder'
                        ' if not given: use input folder')
    parser.add_argument('--out_suffix', default='_lines',
                        help='output suffix (without file extension),'
                        ' filetype will be png')
    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser('lines')
    parser = parseArgs(parser)
    args = parser.parse_args()
    
    if args.indir:
        assert(os.path.exists(args.indir))
    if args.outdir:
        assert(os.path.exists(args.outdir))
        outdir = args.outdir
    else:
        if args.indir:
            outdir = args.indir
        else:
            outdir = os.path.dirname(args.file)
    assert(os.path.exists(args.aru_model))
    assert( (args.scale > 0) and (args.scale <= 1.0) )
    
    if args.file:
        files = [ args.file ]
    else:
        files, _ = getFiles(args.indir, args.suffix, args.labels)
    assert(len(files) > 0)

    infer = Inference_pb(args.aru_model, files, args.scale, 
                         outdir=outdir,
                         out_suffix=args.out_suffix)
    infer.applyARUNet()
