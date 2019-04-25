# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import cv2

class CentroidTracker():
    def __init__(self, video_name, maxDisappeared=5):
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()

        self.frames = OrderedDict();
        self.objectFrameInfo = OrderedDict()
        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = maxDisappeared

        self.padding_scale_factor = 0.10

    def getStatus(self):
        return "Tracking " + str(len(self.objects)) + " objects and " + str(len(self.disappeared)) + " disappeared"

    def register(self, centroid, rect, frame_num):
        # when registering an object we use the next available object
        # ID to store the centroid
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0

        self.frames[frame_num]['object_ids'].append(self.nextObjectID)
        self.objectFrameInfo[self.nextObjectID] = OrderedDict()
        self.objectFrameInfo[self.nextObjectID][frame_num] = rect

        self.nextObjectID += 1

    def deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[objectID]
        del self.disappeared[objectID]

        frames_data = self.objectFrameInfo[objectID]
        frame_height, frame_width = self.frames[list(self.frames.keys())[0]]['frame'].shape[:2]

        print("Deregistering", objectID, ":", frames_data)
        if len(frames_data.keys()) > 10:
            print("Saving video candidate", objectID)
            min_left = 10000000000
            max_right = -1
            min_top = 10000000000
            max_bottom = -1
            for frame_no, box in frames_data.items():
                if box[0] < min_left:
                    min_left = box[0]
                if box[2] > max_right:
                    max_right = box[2]
                if box[1] < min_top:
                    min_top = box[1]
                if box[3] > max_bottom:
                    max_bottom = box[3]

            width_padding = (max_right - min_left) * self.padding_scale_factor
            height_padding = (max_bottom - min_top) * self.padding_scale_factor

            min_left = int(max(0, min_left - width_padding*0.5))
            max_right = int(min(frame_width, max_right + width_padding*0.5))
            min_top = int(max(0, min_top - height_padding*0.5))
            max_bottom = int(min(frame_height, max_bottom + height_padding*0.5))

            max_width = max_right - min_left
            max_height = max_bottom - min_top

            start_frame = list(frames_data.keys())[0]
            end_frame = list(frames_data.keys())[-1]

            # 1. get the centroid and create correct sized boxes
            # 2. write to csv video writer
            # 3. delete frame images
            # 4. interpolate for skipped frames
            out_fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            out = cv2.VideoWriter('output/'+ self.video_name + str(objectID) + '.mp4', out_fourcc, 20, (max_width, max_height))
            for i in range(start_frame, end_frame + 1):
                im_frame = self.frames[i]['frame']
                #if frame_no in frames_data:
                cropped = im_frame[min_top:max_bottom, min_left:max_right]
                out.write(cropped)
                    # cv2.imshow("cropped", cropped)
                    # cv2.waitKey(0)
            out.release()

        del self.objectFrameInfo[objectID]

    def update(self, rects, frame_num, frame_image):
        # store the frame, because at least one object is in it
        self.frames[frame_num] = { 'frame' : frame_image, 'object_ids' : []}

        # check to see if the list of input bounding box rectangles
        # is empty
        if len(rects) == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            # return early as there are no centroids or tracking info
            # to update
            return self.objects

        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        # loop over the bounding box rectangles
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            # use the bounding box coordinates to derive the centroid
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i], rects[i], frame_num)

        # otherwise, are are currently tracking objects so we need to
        # try to match the input centroids to existing object
        # centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value is at the *front* of the index
            # list
            rows = D.min(axis=1).argsort()

            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]


            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()

            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                # val
                if row in usedRows or col in usedCols:
                    continue

                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                self.objectFrameInfo[objectID][frame_num] = rects[col]
                self.frames[frame_num]['object_ids'].append(objectID)

                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)

            # compute both the row and column index we have NOT yet
            # examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                # check to see if the number of consecutive
                # frames the object has been marked "disappeared"
                # for warrants deregistering the object
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

                    # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col], rects[col], frame_num)

        max_frames_to_keep = 1000
        if len(self.frames) > max_frames_to_keep:
            excess_frames = len(self.frames) - max_frames_to_keep
            for i in range(0, excess_frames):
                popped_frame_index = self.frames.popitem(last=False)
                # print("Popped frame ", popped_frame_index)
        # return the set of trackable objects
        return self.objects
