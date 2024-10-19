chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
    switch (request.action) {
        case "scanText":
            sendResponse({data: scanText()});
            break;
    }

});

// Use jQuery
function scanText() {
    let textList = [];
    $("p, h1, h2, h3, h4, h5, h6, blockquote").each(function() {
        textList.push($(this).text());
    });
    console.log(textList);
    return textList;
}
