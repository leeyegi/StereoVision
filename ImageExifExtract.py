#extract image metadata info
#focal length는 (476,100)로 동일
#baseline은 대략 235~245

import PIL.Image
import PIL.ExifTags

img = PIL.Image.open('./dataset_v2/10_02_test_basic.jps')
exif_data = img._getexif()
#print("exif_data")
#print(exif_data)

img2 = PIL.Image.open('./dataset_v2/10_02_test_max.jps')
exif_data = img._getexif()

img3 = PIL.Image.open('./dataset_v2/10_02_test_min.jps')
exif_data = img._getexif()

exif = {
    PIL.ExifTags.TAGS[k]: v
    for k, v in img._getexif().items()
    if k in PIL.ExifTags.TAGS
}

exif2 = {
    PIL.ExifTags.TAGS[k]: v
    for k, v in img2._getexif().items()
    if k in PIL.ExifTags.TAGS
}


exif3 = {
    PIL.ExifTags.TAGS[k]: v
    for k, v in img3._getexif().items()
    if k in PIL.ExifTags.TAGS
}

print("exif")
print(exif)
print("\n")
print("exif2")
print(exif2)
print("\n")
print("exif3")
print(exif3)

