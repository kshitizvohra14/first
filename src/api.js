await axios.post('http://localhost:5000/predict', {
    image: imageSrc // base64 string
  });
  