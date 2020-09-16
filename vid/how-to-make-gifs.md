# How to make HQ gifs with ffmpeg

Based on guide by GIPHY: 
```
https://engineering.giphy.com/how-to-make-gifs-with-ffmpeg/
```

---

## Code:
```
ffmpeg -i <input.gif>
-filter_complex "[0:v] scale=w=800:h=-1,split [a][b];[a] palettegen [p];[b][p] paletteuse,setpts=0.1*PTS,fps=24" 
<output.gif>
```

## Meaning:

Read from the file:

`-i <input.gif>`

Set complex filter (options):

`-filter_complex `

Make and use color palette for gif:

`[0:v] split [a][b];[a] palettegen [p];[b][p] paletteuse`

Resize gif:
[FFMPEG docs on SCALE](https://ffmpeg.org/ffmpeg-filters.html#scale-1)

`scale=w=600:h=-1`

Speed up gif with set framerate:
[FFMPEG docs on SETPTS](https://ffmpeg.org/ffmpeg-filters.html#setpts)
[FFMPEG docs on FPS](https://ffmpeg.org/ffmpeg-filters.html#fps-1)

`setpts=0.1*PTS,fps=24`

Separate additional arguments with commas. 

Arguments before first semicolon applies to input video, while after the last semicolon applies to the output.