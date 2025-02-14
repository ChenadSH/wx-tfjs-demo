// pages/cup/index.js
const app = getApp()

import { Classifier } from '../../../models/coco-ssd/classifier.js'

const CANVAS_ID = 'canvas'

Page({

  classifier: null,

  ctx: null,

  /**
   * 页面的初始数据
   */
  data: {
    cameraBlockHeight: app.globalData.systemInfo.screenHeight - app.globalData.CustomBar,
    predicting: false,
    resultWord: 'null'
  },

  /**
   * 生命周期函数--监听页面加载
   */
  onLoad: function (options) {
    //
  },

  /**
   * 生命周期函数--监听页面初次渲染完成
   */
  onReady: function () {
    setTimeout(() => {
      this.ctx = wx.createCanvasContext(CANVAS_ID)
    }, 500)

    this.initClassifier()

    const context = wx.createCameraContext(this)

    let count = 0
    const listener = context.onCameraFrame((frame) => {
      count = count + 1
      if (count === 10) {
        count = 0
        this.executeClassify(frame)
      }
    })
    listener.start()
  },

  initClassifier() {
    const _this = this
    this.showLoadingToast()
    // console.log(app.globalData.systemInfo.screenWidth,this.data.cameraBlockHeight)
    this.classifier = new Classifier('back', {
      width: app.globalData.systemInfo.screenWidth,
      height: this.data.cameraBlockHeight
    })
    // debugger
    this.classifier.load().then(_ => {
      // console.log(_this.classifier)
      this.hideLoadingToast()
    }).catch(err => {
      console.log(err)
      wx.showToast({
        title: '网络连接异常',
        icon: 'none'
      })
    })
  },

  showLoadingToast() {
    wx.showLoading({
      title: '拼命加载模型',
    })
  },

  hideLoadingToast() {
    wx.hideLoading()
  },

  executeClassify: function (frame) {
    if (this.classifier && this.classifier.isReady() && !this.data.predicting) {
      this.setData({
        predicting: true
      }, () => {
        this.classifier.detect(frame).then((res) => {
          this.classifier.drawBoxes(this.ctx, res)
          this.setData({
            resultWord : res.length,
            predicting: false,
          })
          res = null
          // console.log(res)
        }).catch((err) => {
          console.log(err)
        })
      })
    }
  },

  /**
   * 用户点击右上角分享
   */
  onShareAppMessage: function () {
    return {
      title: 'AI Pocket - 通用物体检测'
    }
  }
})
