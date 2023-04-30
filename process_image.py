import cv2
from neuro.neuroNetwork import NeuroNetwork
import numpy as np
import struct


def nothing(self):
    pass


def process_image(n: NeuroNetwork, size_x: int, size_y: int, shapes_names, init, read, save):

    init()

    # horizontal FOW (deg)
    h_fow = 65
    # vertical FOW (deg)
    v_fow = 65

    # create variables for mask parameters (HSV) with default parameters
    l_h = 0
    l_s = 0
    l_v = 0
    u_h = 179
    u_s = 255
    u_v = 255

    # try to get mask parameters (HSV) from BIN file
    # IMPORTANT: This BIN file must exist even it is empty
    fr = open("HSVb.bin", "rb")
    f_content = fr.read()

    # if BIN file is empty, fill it by default parameters
    if len(f_content) == 0:

        fr.close()

        fw = open("HSVb.bin", "wb")

        fw.write(struct.pack('I', l_h))
        fw.write(struct.pack('I', l_s))
        fw.write(struct.pack('I', l_v))

        fw.write(struct.pack('I', u_h))
        fw.write(struct.pack('I', u_s))
        fw.write(struct.pack('I', u_v))

        fw.close()

    # else read mask parameters (HSV)
    else:
        arr = struct.unpack("I" * (len(f_content) // 4), f_content)

        l_h = arr[0]
        l_s = arr[1]
        l_v = arr[2]

        u_h = arr[3]
        u_s = arr[4]
        u_v = arr[5]

        fr.close()

    # create windows for showing results
    cv2.namedWindow("Live transmission", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Mask", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Filtered mask", cv2.WINDOW_AUTOSIZE)

    # create window for setting mask parameters (HSV)
    cv2.namedWindow("Tracking", cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("LH", "Tracking", l_h, 179, nothing)
    cv2.createTrackbar("LS", "Tracking", l_s, 255, nothing)
    cv2.createTrackbar("LV", "Tracking", l_v, 255, nothing)
    cv2.createTrackbar("UH", "Tracking", u_h, 179, nothing)
    cv2.createTrackbar("US", "Tracking", u_s, 255, nothing)
    cv2.createTrackbar("UV", "Tracking", u_v, 255, nothing)
    cv2.createTrackbar("KS", "Tracking", 7, 7, nothing)

    while True:
        # getting image
        frame = read()

        # convert image from BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # getting mask parameters (HSV) from sliders
        l_h = cv2.getTrackbarPos("LH", "Tracking")
        l_s = cv2.getTrackbarPos("LS", "Tracking")
        l_v = cv2.getTrackbarPos("LV", "Tracking")

        u_h = cv2.getTrackbarPos("UH", "Tracking")
        u_s = cv2.getTrackbarPos("US", "Tracking")
        u_v = cv2.getTrackbarPos("UV", "Tracking")

        kernel_size = cv2.getTrackbarPos("KS", "Tracking")

        # if lower hue border is bigger than upper hue border, make two regions and merge them
        if l_h > u_h:
            l1_b = np.array([l_h, l_s, l_v])
            u1_b = np.array([179, u_s, u_v])

            l2_b = np.array([0, l_s, l_v])
            u2_b = np.array([u_h, u_s, u_v])

            mask = cv2.inRange(hsv, l1_b, u1_b) + cv2.inRange(hsv, l2_b, u2_b)

        # else do normal
        else:
            l_b = np.array([l_h, l_s, l_v])
            u_b = np.array([u_h, u_s, u_v])

            mask = cv2.inRange(hsv, l_b, u_b)

        # define kernel
        kernel = np.array([[ 0, -1, -1, -1,  0],
                           [-1,  1,  1,  1, -1],
                           [-1,  1,  5,  1, -1],
                           [-1,  1,  1,  1, -1],
                           [ 0, -1, -1, -1,  0]])

        # filter mask from noises
        #filtered_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        filtered_mask = cv2.filter2D(src=mask, ddepth=-1, kernel=kernel)

        # find contours
        contours = cv2.findContours(filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        # if there is few contours select biggest one
        if len(contours) > 1:
            c = max(contours, key=cv2.contourArea)
        elif len(contours) == 1:
            c = contours[0]

        res = frame.copy()

        c_exist = False
        if len(contours) != 0:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(res, (x, y), (x + w, y + h), (0, 255, 0), 4)
            c_exist = True

        cv2.imshow("Live transmission", res)
        cv2.imshow("Mask", mask)
        cv2.imshow("Filtered mask", filtered_mask)

        # wait for key 5 ms
        key = cv2.waitKey(5)

        # if key is S, save mask parameters from sliders to BIN file
        if key == ord('s'):

            fw = open("HSVb.bin", "wb")

            fw.write(struct.pack('I', l_h))
            fw.write(struct.pack('I', l_s))
            fw.write(struct.pack('I', l_v))

            fw.write(struct.pack('I', u_h))
            fw.write(struct.pack('I', u_s))
            fw.write(struct.pack('I', u_v))

            fw.close()
            print("Mask parameters were saved")
            cv2.waitKey(1500)

        # if key is R, calculate position and send to ESP32 CAM
        elif key == ord('r') and c_exist:

            # getting size of image
            size_fx = len(frame[0])
            size_fy = len(frame)

            # position calculation
            ang_x = (x + w/2 - size_fx/2)/size_fx * h_fow
            ang_y = (y + h/2 - size_fy/2)/size_fy * v_fow

            print(f'Position: {ang_x} deg, {ang_y} deg')

            print(f'Mask shape: {mask.shape}')
            print(f'Filtered mask shape: {filtered_mask.shape}')

            if w < size_x or h < size_y:
                print('Object is too little')
            else:
                try:
                    crop = mask[y:y + h, x:x + w]
                    cv2.imshow("Cropped", crop)
                    resized = cv2.resize(crop, (size_x, size_y))
                    cv2.imshow("Resized", cv2.resize(resized, (300, 300)))
                    shape = n.predict(np.matrix((resized / 255.0).flatten()).T, first_element_is_one=False)

                    print(f'Shape: ')
                    for i in range(len(list(shape))):
                        print(f'  {shapes_names[i]}: {shape[i,0]:.1%}')

                    last_image = crop.copy()

                except Exception as e:
                    print(f'Oops... ({str(e)})')

            cv2.waitKey(1500)

        # if key is T, save last sample as true
        elif key == ord('t'):
            save(last_image, True)

        # if key is F, save last sample as false
        elif key == ord('f'):
            save(last_image, False)

        # if key is Q, quit program
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
