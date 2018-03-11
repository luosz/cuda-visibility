# Voreen Python script
import voreen
import voreenqt

voreen.info()
voreenqt.info()

from os import listdir
from os.path import isfile, join
mypath="D:\\document\\work\\time-varying-visualization\\run"
tfs = [f for f in listdir(mypath) if isfile(join(mypath, f)) and -1!=f.find(".tfi")]
print(tfs)

for f in tfs:
    tf_filename=join(mypath,f)
    voreen.loadTransferFunction("SingleVolumeRaycaster", "transferFunction", tf_filename)
    voreen.repaint()

    image_filename=tf_filename.replace(".tfi",".png")
    voreen.snapshotCanvas(0, image_filename)
    print("saved to image %s" % image_filename)
