# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import cv2 as cv
from django.shortcuts import render
from django.http import HttpResponse
from django.views.generic.base import View

# Create your views here.

framework = 'darknet'
config = 'yolov3-tiny.cfg'
model = 'yolov3-tiny.weights'

def callback(pos):
    global confThreshold
    confThreshold = pos / 100.0

net = cv.dnn.readNet(model, config, framework)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

classes = None
confThreshold = 0.5

def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    def drawPred(classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0))

        label = '%.2f' % conf

        # Print a label of class.
        if classes:
            assert(classId < len(classes))
            label = '%s: %s' % (classes[classId], label)

        labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255), cv.FILLED)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    layerNames = net.getLayerNames()
    lastLayerId = net.getLayerId(layerNames[-1])
    lastLayer = net.getLayer(lastLayerId)

    if net.getLayer(0).outputNameToIndex('im_info') != -1:  # Faster-RCNN or R-FCN
        # Network produces output blob with a shape 1x1xNx7 where N is a number of
        # detections and an every detection is a vector of values
        # [batchId, classId, confidence, left, top, right, bottom]
        assert(len(outs) == 1)
        out = outs[0]
        for detection in out[0, 0]:
            confidence = detection[2]
            if confidence > confThreshold:
                left = int(detection[3])
                top = int(detection[4])
                right = int(detection[5])
                bottom = int(detection[6])
                classId = int(detection[1]) - 1  # Skip background label
                drawPred(classId, confidence, left, top, right, bottom)
    elif lastLayer.type == 'DetectionOutput':
        # Network produces output blob with a shape 1x1xNx7 where N is a number of
        # detections and an every detection is a vector of values
        # [batchId, classId, confidence, left, top, right, bottom]
        assert(len(outs) == 1)
        out = outs[0]
        for detection in out[0, 0]:
            confidence = detection[2]
            if confidence > confThreshold:
                left = int(detection[3] * frameWidth)
                top = int(detection[4] * frameHeight)
                right = int(detection[5] * frameWidth)
                bottom = int(detection[6] * frameHeight)
                classId = int(detection[1]) - 1  # Skip background label
                drawPred(classId, confidence, left, top, right, bottom)
    elif lastLayer.type == 'Region':
        # Network produces output blob with a shape NxC where N is a number of
        # detected objects and C is a number of classes + 4 where the first 4
        # numbers are [center_x, center_y, width, height]
        classIds = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = center_x - width / 2
                    top = center_y - height / 2
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
        indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, 0.4)
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            drawPred(classIds[i], confidences[i], left, top, left + width, top + height)


def run_detection(file):
    image = Image.open(file.file)
    matrix = np.array(image)
    frame = matrix[:, : ::-1].copy()
    
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Create a 4D blob from a frame.
    inpWidth = args.width if args.width else frameWidth
    inpHeight = args.height if args.height else frameHeight
    blob = cv.dnn.blobFromImage(frame, args.scale, (inpWidth, inpHeight), args.mean, args.rgb, crop=False)

    # Run a model
    net.setInput(blob)

    outs = net.forward(getOutputsNames(net))
    postprocess(frame, outs)

class DetectionView(View):
    def get(self, request):
	return HttpResponse('How did you find me?')

    def post(self, request):
	file = request.FILES['file']
	image = run_detection(file)

	return stored_name
    

