chrome.runtime.onMessage.addListener(async function(request, sender, sendResponse) {
    switch (request.action) {
        case "scanText":
            const textData = await scanText();
            sendResponse({data: textData});
            break;
        case "scanImages":
            sendResponse({images: scanImages()});
            break;
        case "scanImage":
            const scanData = await scanImage(request.imgUrl);
            sendResponse({data: scanData});
            break;
    }
});

async function scanText() {
    let textList = [];
    let elements = [];

    $("p, h1, h2, h3, h4, h5, h6, blockquote").each(function() {
        if ($(this).text().length < 80) {
            return;
        }
        textList.push($(this).text());
        elements.push($(this));  // Store the jQuery element to style it later
    });

    // Send the collected text to the server
    try {
        const response = await fetch('https://example.com/scanText', { // Replace with actual server URL
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ texts: textList })
        });

        const data = await response.json();

        // Assume `data` contains an array of scores corresponding to the text blocks
        data.scores.forEach((fakeScore, index) => {
            if (fakeScore < 0.2) {
                elements[index].css(borderCSS("#A3DDCB"));
            } else if (fakeScore < 0.5) {
                elements[index].css(borderCSS("#E8E9A1"));
            } else if (fakeScore < 0.8) {
                elements[index].css(borderCSS("#E6B566"));
            } else {
                elements[index].css(borderCSS("#E5707E"));
            }
        });

        return data;  // Return the response from the server

    } catch (error) {
        console.error("Error sending text to the server:", error);
        return { error: "Failed to send text to the server" };
    }
}

function scanImages() {
    let images = [];
    $("img").each(function() {
        images.push($(this).attr("src"));
    });
    return images;
}

// Modified scanImage to send image URL to the server and get a fake score
async function scanImage(targetUrl) {
    try {
        const response = await fetch('https://example.com/scanImage', {  // Replace with actual server URL
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ imageUrl: targetUrl })
        });

        const data = await response.json();

        // TODO: Change this to the actual key used in the response
        const fakeScore = data.fakeScore;
        console.log("Image scan result:", fakeScore);
        return fakeScore;

    } catch (error) {
        console.error("Error sending image to the server:", error);
    }
}

function borderCSS(colour) {
    return {
        "box-shadow": `-4px 0px 1px -1px ${colour}`,
        "padding-left": "10px",
        "margin-left": "-10px"
    }
}
