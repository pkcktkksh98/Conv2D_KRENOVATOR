import { useRef,useState } from 'react';
import CanvasDraw from 'react-canvas-draw';
import axios from 'axios';
function App(){
    const ref =useRef<CanvasDraw>(null);
    const [predictedNumber, setPredictedNumber]=useState<number>(0);
    const requestPredication = async(base64:string)=>{
        const URL_HERE = "http://localhost:5000/predict";

        const res = await axios.post(URL_HERE,{
            image:base64.split(",")[1],
        });

        if(res.data?.length){
            console.log(res.data[0]);
            setPredictedNumber(res.data[0]);
        }
    }
    const onClear = ()=>{
        if(ref.current){
            ref.current.clear();
        }
        
    }
    const onClickSend = ()=>{
        if(ref.current){
            const imageURL = ref.current.getDataURL("png",false,0xffffff);
            requestPredication(imageURL)

        }
    }
    return <div style={{display:'flex', flexDirection:'column', padding:5}}>
                <h2 style={{lineHeight:0}}>Draw your number here:</h2>
                {/*Canvas*/}
                <CanvasDraw
                    canvasHeight={280}
                    canvasWidth={280}
                    ref={ref}
                />
                <div>
                    <button onClick={()=>{onClear();}}> Clear</button>
                    <button onClick={()=>{onClickSend();}}>Send</button>
                </div>

                <div>
                    Predicted Number:{predictedNumber}
                </div>
            </div>;
}

export default App;