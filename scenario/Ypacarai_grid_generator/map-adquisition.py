import numpy as np 
import cv2
import sys, getopt
import os.path
from numpy import savetxt

def nothing(x):
   pass 
argv = sys.argv[1:]

try:
    opts, args = getopt.getopt(argv,"dc")
    
except:
    print(f"\nProgram for loading and transforming a map into a binary mask")
    print(f"\nFormat: python map-adquisition.py [MODE] [PATH_TO_IMAGE]\n")
    print(f"\nModes:\n")
    print(f"\t-d : default mode with normal Ypacarai parameters")
    print(f"\t-c : calibration mode for other cases\n")
    exit()

default_mode = 0

for opt,args in opts:
    if opt == '-d':
        print(f"Selecting default mode")
        default_mode = 1
    elif opt in "-c":
        print(f"Selecting calibration mode")
        default_mode = 0
    else:
        exit()

#上传图片#
img = cv2.imread('YpacarayMap_color.png')

scale_percent = 50 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height) 
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

img = cv2.resize(img,(240,240))

# 去HSV空间带出颜色#
hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

if(default_mode == 0):
    # 创建用于校准的滑块#
    cv2.namedWindow('Parametros')
    cv2.createTrackbar('Minimal hue','Parametros',0,179,nothing)
    cv2.createTrackbar('Maximum hue','Parametros',0,179,nothing)
    cv2.createTrackbar('Minimum saturation','Parametros',0,255,nothing)
    cv2.createTrackbar('Maximum saturation','Parametros',0,255,nothing)
    cv2.createTrackbar('Minial value','Parametros',0,255,nothing)
    cv2.createTrackbar('Maximum value','Parametros',0,255,nothing)

while(True):

    if(default_mode == 0):
        #读取滑块并保存H,S,V的值以构建范围：
        hMin = cv2.getTrackbarPos('Minimal hue','Parametros')
        hMax = cv2.getTrackbarPos('Maximum hue','Parametros')
        sMin = cv2.getTrackbarPos('Minimum saturation','Parametros')
        sMax = cv2.getTrackbarPos('Maximum saturation','Parametros')
        vMin = cv2.getTrackbarPos('Minimal value','Parametros')
        vMax = cv2.getTrackbarPos('Maximum value','Parametros')

        #创建定义颜色范围的数组：
        lower_tresh = np.array([hMin,sMin,vMin])
        upper_tresh=np.array([hMax,sMax,vMax])
    
    else:
        lower_tresh = np.array([82,109,150])
        upper_tresh = np.array([125,154,222])

    # 结果掩码 #
    mask = cv2.inRange(hsv_img, lower_tresh, upper_tresh)
    mask_filtered = cv2.medianBlur(mask,int(scale_percent/100*15))

    # 将掩码裁剪到原始地图上
    crop = cv2.bitwise_and(img, img, mask=mask_filtered)

    # 显示结果并退出：
    cv2.imshow('Original',img)
    cv2.imshow('Mascara',mask_filtered)
    cv2.imshow('Superposicion',crop)


    # 等待退出#
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.imwrite('MASK.png',mask_filtered)

# 保存结果


# 进行掩码重采样 #

ancho_mapa_m = 15000 # 米 #
alto_mapa_m = 13000 # 米 #

resolucion_m_each_cell = 500 # 每个单元格占用resolucion_m_each_cell米 #

width_scale = round(ancho_mapa_m/resolucion_m_each_cell)
height_scale = round(alto_mapa_m/resolucion_m_each_cell)

new_size = (width_scale,height_scale)
grid_map =  cv2.resize(mask_filtered,new_size,interpolation=cv2.INTER_NEAREST)

# 使用重采样的地图，进行裁剪以减少非可能单元格的数量 #

x_f,y_f,w_f,h_f = cv2.boundingRect(grid_map)

print(f"The map will be rescaled with {width_scale} cells high and {height_scale} cells wide")

celdas_AGUA = cv2.countNonZero(grid_map)
celdas_total = width_scale*height_scale
celdas_TIERRA = celdas_total-celdas_AGUA

print(f"In total, we will have {celdas_AGUA} WATER cells (WHITE) and {celdas_TIERRA} LAND cells (BLACK). {celdas_total} cells in total")

cropped_image = np.zeros((h_f+1,w_f+1))
cropped_image = grid_map[y_f:y_f+h_f,x_f:x_f+w_f]/255
cropped_image = cv2.copyMakeBorder(cropped_image,1,1,1,1, borderType = cv2.BORDER_CONSTANT, value = 0)

cv2.namedWindow('Mapa de celdas',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Mapa de celdas', 600,600)
cv2.imshow('Mapa de celdas',cropped_image )

while(True):
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()

if(default_mode == 1):
	savetxt('YpacaraiMap.csv', cropped_image , delimiter=',', fmt = '%u')
	print(f"CSV file created!")
else:
	savetxt('YpacaraiMap_calibrated.csv', cropped_image , delimiter=',', fmt = '%u')
	print(f"CSV file created!")
