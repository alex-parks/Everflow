class ImageCache {
  constructor(maxSize = 100) {
    this.cache = new Map()
    this.maxSize = maxSize
    this.loadingPromises = new Map()
  }

  _generateKey(sequenceId, frameNum, exposure = 0.0) {
    return `${sequenceId}_${frameNum}_${exposure.toFixed(2)}`
  }

  _evictLRU() {
    if (this.cache.size >= this.maxSize) {
      // Remove oldest entries (first added)
      const keysToRemove = Array.from(this.cache.keys()).slice(0, this.cache.size - this.maxSize + 10)
      keysToRemove.forEach(key => {
        const item = this.cache.get(key)
        if (item && item.url) {
          URL.revokeObjectURL(item.url)
        }
        this.cache.delete(key)
      })
    }
  }

  async loadImage(sequenceId, frameNum, exposure = 0.0) {
    const key = this._generateKey(sequenceId, frameNum, exposure)
    
    // Return cached image if available
    if (this.cache.has(key)) {
      const item = this.cache.get(key)
      // Move to end (LRU)
      this.cache.delete(key)
      this.cache.set(key, item)
      return item.url
    }

    // Check if already loading
    if (this.loadingPromises.has(key)) {
      return this.loadingPromises.get(key)
    }

    // Start loading
    const loadPromise = this._fetchAndCacheImage(sequenceId, frameNum, exposure, key)
    this.loadingPromises.set(key, loadPromise)

    try {
      const url = await loadPromise
      this.loadingPromises.delete(key)
      return url
    } catch (error) {
      this.loadingPromises.delete(key)
      throw error
    }
  }

  async _fetchAndCacheImage(sequenceId, frameNum, exposure, key) {
    try {
      const response = await fetch(
        `http://localhost:4005/api/frames/${sequenceId}/frame/${frameNum}?exposure=${exposure}`
      )
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const blob = await response.blob()
      const url = URL.createObjectURL(blob)

      // Cache the image
      this._evictLRU()
      this.cache.set(key, {
        url,
        timestamp: Date.now(),
        sequenceId,
        frameNum,
        exposure
      })

      return url
    } catch (error) {
      console.error(`Error loading frame ${frameNum}:`, error)
      throw error
    }
  }

  async preloadFrames(sequenceId, frameNumbers, exposure = 0.0) {
    const promises = frameNumbers.map(frameNum => {
      const key = this._generateKey(sequenceId, frameNum, exposure)
      if (!this.cache.has(key) && !this.loadingPromises.has(key)) {
        return this.loadImage(sequenceId, frameNum, exposure).catch(error => {
          console.warn(`Failed to preload frame ${frameNum}:`, error)
        })
      }
      return Promise.resolve()
    })

    return Promise.allSettled(promises)
  }

  async preloadRange(sequenceId, startFrame, endFrame, exposure = 0.0) {
    const frameNumbers = []
    for (let i = startFrame; i <= endFrame; i++) {
      frameNumbers.push(i)
    }
    return this.preloadFrames(sequenceId, frameNumbers, exposure)
  }

  async preloadAdjacent(sequenceId, currentFrame, totalFrames, exposure = 0.0, radius = 5) {
    const framesToPreload = []
    
    // Preload frames around current frame
    for (let i = -radius; i <= radius; i++) {
      const frameNum = currentFrame + i
      if (frameNum >= 0 && frameNum < totalFrames && frameNum !== currentFrame) {
        framesToPreload.push(frameNum)
      }
    }

    return this.preloadFrames(sequenceId, framesToPreload, exposure)
  }

  clearSequence(sequenceId) {
    const keysToRemove = []
    for (const [key, item] of this.cache.entries()) {
      if (item.sequenceId === sequenceId) {
        keysToRemove.push(key)
        if (item.url) {
          URL.revokeObjectURL(item.url)
        }
      }
    }
    keysToRemove.forEach(key => this.cache.delete(key))
  }

  clearAll() {
    for (const [key, item] of this.cache.entries()) {
      if (item.url) {
        URL.revokeObjectURL(item.url)
      }
    }
    this.cache.clear()
    this.loadingPromises.clear()
  }

  getCacheStats() {
    return {
      size: this.cache.size,
      maxSize: this.maxSize,
      loadingCount: this.loadingPromises.size
    }
  }

  // Initiate backend preloading for a range
  async requestBackendPreload(sequenceId, startFrame, endFrame, exposure = 0.0) {
    try {
      const url = new URL(`http://localhost:4005/api/frames/${sequenceId}/preload`)
      url.searchParams.append('start_frame', startFrame)
      url.searchParams.append('end_frame', endFrame)
      url.searchParams.append('exposure', exposure)
      
      await fetch(url, {
        method: 'POST'
      })
    } catch (error) {
      console.warn('Backend preload request failed:', error)
    }
  }
}

// Global cache instance
export const imageCache = new ImageCache(150) // Cache up to 150 images

export default ImageCache