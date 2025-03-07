import cv2

img = cv2.imread('uploads/20240524173053.png', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (6000, 8000))
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# 二值化
_, binary_img = cv2.threshold(img, 135, 255, cv2.THRESH_BINARY)
cnts, _ = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for contour in cnts:
    x, y, width, height = cv2.boundingRect(contour)
    ar = width / float(height)
    # 筛选合适轮廓
    if 2.3 <= ar <= 2.7 and 3000 <= width <= 4200 and 1000 <= height <= 1800:
        table_img = img[y - 50:y + height + 50, x - 50:x + width + 50]
        table_img = cv2.resize(table_img, (1150, 500), interpolation=cv2.INTER_AREA)
        cv2.imwrite('uploads/answer_form_test.png', table_img)
