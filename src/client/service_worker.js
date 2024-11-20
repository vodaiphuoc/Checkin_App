const addResourcesToCache = async (resources) => {
  const cache = await caches.open("v2.1");
  await cache.addAll(resources);
};


// Service worker logic
self.addEventListener('install', function(event) {
  event.waitUntil(
    caches.open('my-pwa-cache').then(function(cache) {
      return cache.addAll([
        '/',
        '/checkin',
        '/login',
        '/signup'
      ]);
    })
  );
});

self.addEventListener('fetch', function(event) {
  event.respondWith(
    caches.match(event.request).then(function(response) {
      return response || fetch(event.request);
    })
  );
});


const deleteCache = async (key) => {
  await caches.delete(key);
};

const deleteOldCaches = async () => {
  const cacheKeepList = ["v2.1"];
  const keyList = await caches.keys();
  const cachesToDelete = keyList.filter((key) => !cacheKeepList.includes(key));
  await Promise.all(cachesToDelete.map(deleteCache));
};

self.addEventListener("activate", (event) => {
  event.waitUntil(deleteOldCaches());
});
