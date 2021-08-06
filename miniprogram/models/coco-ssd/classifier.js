import * as tf from '@tensorflow/tfjs-core'
import * as tfc from '@tensorflow/tfjs-converter'
import '@tensorflow/tfjs-backend-cpu'
import * as cocoSsd from '@tensorflow-models/coco-ssd'

import { getFrameSliceOptions } from '../utils/util'

const fontSize = 16
const color = 'aqua'
const lineWidth = 2

// 以下两个为 SSD 模型的地址，请下载放到自己的服务器，然后修改以下链接
const SSD_NET_URL = 'https://h5.dayanweb.cn/cxTest/tfmodel/cup/model.json'
// const SSD_NET_URL = 'https://ai.flypot.cn/models/coco-ssd/model.json'
// const SSD_NET_BIN_URL = 'https://ai.flypot.cn/models/mobilenet/group1-shard1of1'
const img_size = 224
// 准确率大于多少显示
const threshold = 0.7;
// 所有标签分类
let classesDir = {
  1: {
      name: 'GreyCup',
      id: 1,
  },
  2: {
      name: 'Other',
      id: 2,
  }
}

export class Classifier {
  // 指明前置或后置 front|back
  cameraPosition

  // 图像显示尺寸结构体 { width: Number, height: Number }
  displaySize

  // 神经网络模型
  ssdNet

  // ready
  ready

  constructor(cameraPosition, displaySize) {
    this.cameraPosition = cameraPosition

    this.displaySize = {
      width: displaySize.width,
      height: displaySize.height
    }

    this.ready = false
  }

  load() {
    return new Promise((resolve, reject) => {
      tfc.loadGraphModel( SSD_NET_URL
      ).then(model => {
        console.log(model)
        this.ssdNet = model
        this.ready = true
        resolve()
      }).catch(err => {
        reject(err)
      })
    })
  }

  isReady() {
    return this.ready
  }

  detect(frame) {
    const _this = this
    return new Promise((resolve, reject) => {
      let tensor = tf.tidy(() => {
        const temp = tf.browser.fromPixels({
          data: new Uint8Array(frame.data),
          width: frame.width,
          height: frame.height,
        }, 3)
        const sliceOptions = getFrameSliceOptions(this.cameraPosition, frame.width, frame.height, this.displaySize.width, this.displaySize.height)
        // tf.image.resizeBilinear(temp,[img_size, img_size]) //tf.reshape(temp,[1, ...temp.shape])
        // return tf.slice(temp, [0, 0, 0], [-1, -1, 3])
        return  tf.image.resizeBilinear(tf.slice(temp, [0, 0, 0], [-1, -1, 3]),[this.displaySize.height, this.displaySize.width])// temp.slice(sliceOptions.start, sliceOptions.size).resizeBilinear([this.displaySize.height, this.displaySize.width]).asType('int32')
      })
      // debugger
      tensor = tf.cast(tensor,'int32')
      tensor = tf.reshape(tensor,[1,...tensor.shape])//...tensor.shape [508,320,3]
      console.log(tensor.shape)
      // console.log(tensor)
      // console.log(tensor.toString())
      // const res = this.ssdNet.executeAsync(tensor)
      // console.log(res)
      // this.ssdNet.predict(tensor).then(res => {
      //   tensor.dispose()
      //   resolve(res)
      // }).catch(err => {
      //   console.log(err)
      //   tensor.dispose()
      //   reject()
      // })
      // const outputDim = [
      //   'num_detections', 'detection_boxes', 'detection_scores',
      //   'detection_classes'
      // ];{image_tensor:tensor}
      this.ssdNet.executeAsync(tensor).then(res => {//,'detection_scores:0'
        tensor.dispose()
        console.log(res)
        // ['num_detections', 'detection_boxes', 'detection_classes', 'detection_scores', 'raw_detection_boxes', 'raw_detection_scores, 'detection_anchor_indices', 'detection_multiclass_scores'] 
        //Getting predictions
        const boxes = res[0].arraySync();
        const scores = res[5].arraySync();//res[3].arraySync();//
        const classes = res[6].dataSync();
        // console.log(classes)
        // console.log(boxes)
        // console.log(scores)
        // console.log(res.print())
        // const r = res.arraySync()
        // console.log(r)

        const detections = _this.buildDetectedObjects(scores, threshold,boxes, classes, classesDir);
          console.log(detections)
        resolve(detections)
      }).catch(err => {
        console.log(err)
        tensor.dispose()
        reject()
      })

    })
  }
  buildDetectedObjects(scores, threshold, boxes, classes, classesDir) {
    const detectionObjects = []
    // var video_frame = document.getElementById('frame');

    // for (let i=0; i<detection_scores.length; i++) {
    //   const score = detection_scores[i];
    //   if (score < 0.5) break; // 置信度过滤
    //   // [ymin , xmin , ymax , xmax]
    //   boxes.push({
    //     ymin: modelOut['detection_boxes'][i*4]*h,
    //     xmin: modelOut['detection_boxes'][i*4+1]*w,
    //     ymax: modelOut['detection_boxes'][i*4+2]*h,
    //     xmax: modelOut['detection_boxes'][i*4+3]*w,
    //     label: labelMap[modelOut['detection_classes'][i]],
    //   })
    // }
    // console.log(boxes)

    classes.forEach((score, i) => {
      if (score > threshold) {
        const bbox = [];
        const minY = boxes[0][i][0] * this.displaySize.height // video_frame.offsetHeight;
        const minX = boxes[0][i][1] * this.displaySize.width // video_frame.offsetWidth;
        const maxY = boxes[0][i][2] * this.displaySize.height // video_frame.offsetHeight;
        const maxX = boxes[0][i][3] * this.displaySize.width // video_frame.offsetWidth;
        bbox[0] = minX;
        bbox[1] = minY;
        bbox[2] = maxX - minX;
        bbox[3] = maxY - minY;
        detectionObjects.push({
          class: 'GreyCup',
          // label: classesDir[classes[i]].name,
          score: score.toFixed(4),
          bbox: bbox
        })
      }
    })
    return detectionObjects
  }
  drawBoxes(ctx, boxes) {
    if (!ctx && !boxes) {
      return
    }

    const minScore = 0.3

    ctx.setFontSize(fontSize)
    ctx.strokeStyle = color
    ctx.lineWidth = lineWidth

    boxes.forEach(box => {
      if (box.score >= minScore) {
        ctx.rect(...(box.bbox))
        ctx.stroke()
        ctx.setFillStyle(color)
        ctx.fillText(box['class'], box.bbox[0], box.bbox[1] - 5)
      }
    })

    ctx.draw()
    return true
  }

  dispose() {
    this.ssdNet.dispose()
  }
}