chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
    if (request.action == 'scanPage') {
        // TODO: Scan page
        sendResponse({status: 'ok'});
    }
});