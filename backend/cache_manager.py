"""Frame cache manager for efficient sequence playback"""
import hashlib
from typing import Optional, Dict, Any
import psutil
import time
import threading

class FrameCache:
    def __init__(self, max_memory_mb: int = 512):
        """Initialize frame cache with memory limit"""
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.current_memory_usage = 0
        self.lock = threading.RLock()
        
    def _make_key(self, sequence_id: str, frame_num: int, exposure: float, size: Optional[int] = None) -> str:
        """Create cache key"""
        if size:
            return f"{sequence_id}:{frame_num}:{exposure}:{size}"
        return f"{sequence_id}:{frame_num}:{exposure}"
    
    def get(self, sequence_id: str, frame_num: int, exposure: float, size: Optional[int] = None) -> Optional[bytes]:
        """Get cached frame data"""
        with self.lock:
            key = self._make_key(sequence_id, frame_num, exposure, size)
            
            if key in self.cache:
                self.access_times[key] = time.time()
                return self.cache[key]['data']
            
            return None
    
    def put(self, sequence_id: str, frame_num: int, exposure: float, data: bytes, size: Optional[int] = None):
        """Cache frame data"""
        with self.lock:
            key = self._make_key(sequence_id, frame_num, exposure, size)
            data_size = len(data)
            
            # Check if we need to evict old entries
            while (self.current_memory_usage + data_size) > self.max_memory_bytes and self.cache:
                self._evict_lru()
            
            # Store the data
            self.cache[key] = {
                'data': data,
                'size': data_size,
                'timestamp': time.time()
            }
            self.access_times[key] = time.time()
            self.current_memory_usage += data_size
    
    def _evict_lru(self):
        """Evict least recently used item"""
        if not self.access_times:
            return
            
        # Find the least recently used key
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        
        # Remove from cache
        if lru_key in self.cache:
            self.current_memory_usage -= self.cache[lru_key]['size']
            del self.cache[lru_key]
        
        del self.access_times[lru_key]
    
    def clear_sequence_cache(self, sequence_id: str):
        """Clear all cached frames for a sequence"""
        with self.lock:
            keys_to_remove = [key for key in self.cache.keys() if key.startswith(f"{sequence_id}:")]
            
            for key in keys_to_remove:
                if key in self.cache:
                    self.current_memory_usage -= self.cache[key]['size']
                    del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]
    
    def clear_all_cache(self):
        """Clear entire cache"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.current_memory_usage = 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            try:
                process = psutil.Process()
                memory_info = process.memory_info()
            except:
                memory_info = None
            
            return {
                "cached_frames": len(self.cache),
                "cache_memory_usage_mb": round(self.current_memory_usage / (1024 * 1024), 2),
                "cache_memory_limit_mb": round(self.max_memory_bytes / (1024 * 1024), 2),
                "cache_hit_ratio": "N/A",  # Would need request counting to implement
                "process_memory_mb": round(memory_info.rss / (1024 * 1024), 2) if memory_info else "N/A",
                "sequences_cached": len(set(key.split(':')[0] for key in self.cache.keys()))
            }

# Global cache instance
frame_cache = FrameCache(max_memory_mb=512)