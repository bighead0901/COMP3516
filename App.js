import { StatusBar } from 'expo-status-bar';
import * as React from 'react';
import { StyleSheet, View, ImageBackground } from 'react-native';
import { BottomNavigation, Text, IconButton, MD3Colors } from 'react-native-paper';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { Audio } from 'expo-av';
import { API } from 'aws-amplify';
import axios from 'axios';
import * as FileSystem from "expo-file-system";


const apiURL = "http://10.0.2.2:5000/api/audio"

const RecordRoute = () => {
    const [recording, setRecording] = React.useState();
    const [predictionResult, setpredictionResult] = React.useState();
    const [recordingDuration, setRecordingDuration] = React.useState("00:00:00.0");
    function setTime(time) {
        //console.log(time)
        setRecordingDuration(msToTime(time.durationMillis))
    }

    function msToTime(duration) {
      var milliseconds = Math.floor((duration % 1000) / 100),
        seconds = Math.floor((duration / 1000) % 60),
        minutes = Math.floor((duration / (1000 * 60)) % 60),
        hours = Math.floor((duration / (1000 * 60 * 60)) % 24);

      hours = (hours < 10) ? "0" + hours : hours;
      minutes = (minutes < 10) ? "0" + minutes : minutes;
      seconds = (seconds < 10) ? "0" + seconds : seconds;

      return hours + ":" + minutes + ":" + seconds + "." + milliseconds;
    }

    async function startRecording() {
        try {
            console.log('Requesting permissions..');
            setpredictionResult();
            await Audio.requestPermissionsAsync();
            await Audio.setAudioModeAsync({
                allowsRecordingIOS: true,
                playsInSilentModeIOS: true,
            });

            console.log('Starting recording..');
            const options = {
              isMeteringEnabled: true,
              android: {
                extension: '.m4a',
                sampleRate: 44100,
                numberOfChannels: 2,
                bitRate: 128000,
              },
              ios: {
                extension: '.m4a',
                sampleRate: 44100,
                numberOfChannels: 2,
                bitRate: 128000,
                linearPCMBitDepth: 16,
                linearPCMIsBigEndian: false,
                linearPCMIsFloat: false,
              },
              web: {
                mimeType: 'audio/webm',
                bitsPerSecond: 128000,
              },
            };

            const { recording } = await Audio.Recording.createAsync(
                options,
                (status) => setTime(status),
                20
            );
            setRecording(recording);
            console.log('Recording started');
        } catch (err) {
            console.error('Failed to start recording', err);
        }
    }

    async function stopRecording() {
        try{
            setpredictionResult("Processing...");
            console.log('Stopping recording..' + recordingDuration);
            setRecording(undefined);
            await recording.stopAndUnloadAsync();
            await Audio.setAudioModeAsync({
                allowsRecordingIOS: false,
            });
            const uri = recording.getURI();
            console.log('Recording stopped and stored at', uri, recording);
            //const response = await API.get('comp3516api', '/');
            try {
                const response = await FileSystem.uploadAsync(apiURL,uri);
                //const body = JSON.parse(response.body);
                let label = "Not recognised... Try again."
                if (response.body == "0") label = "It's a male human voice!"
                else if (response.body == "1") label = "It's a male robotic voice!"
                else if (response.body == "2") label = "It's a female human voice!"
                else if (response.body == "3") label = "It's a female robotic voice!"
                else label = "Error in processing.. Try again later."
                setpredictionResult(label);
            } catch (err) {
                console.error(err);
            }

        }catch (e){
            console.error(e)
        }
    }

    return(
        <ImageBackground source={require('./src/bg.png')} resizeMode="cover" style={styles.image}>
            <View style={styles.container}>
                <Text style={styles.text}>Speech Liveness Detection</Text>
                <Text style={styles.text2}>
                    {recording ? `Recording... ${recordingDuration}` : 'Press button below to start recording'}
                </Text>
            </View>
            {predictionResult ? <Text style={styles.text2}>
                {predictionResult}
            </Text>: null}
            <IconButton
                icon="record"
                iconColor={recording ? "red" : "grey"}
                mode='outlined'
                size={50}
                onPress={recording ? stopRecording : startRecording}
                style={{marginBottom: 50}}
            />
        </ImageBackground>

    )
}

const FilesRoute = () => {
    return(
        <ImageBackground source={require('./src/bg.png')} resizeMode="cover" style={styles.image}>
        </ImageBackground>
    )
}

export default function App() {
        const [sound, setSound] = React.useState();
    const [index, setIndex] = React.useState(0);
    const [routes] = React.useState([
    { key: 'record', title: 'Recording', focusedIcon: 'record-circle', unfocusedIcon: 'record-circle-outline'},
    { key: 'files', title: 'History', focusedIcon: 'file', unfocusedIcon: 'file-outline' },
    ]);

    const renderScene = BottomNavigation.SceneMap({
        record: RecordRoute,
        files: FilesRoute,
    });

    return (
        <SafeAreaProvider>
            <RecordRoute/>
            {/*<BottomNavigation
                navigationState={{ index, routes }}
                onIndexChange={setIndex}
                renderScene={renderScene}
            />*/}
        </SafeAreaProvider>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        marginTop: 180,
        marginBottom: 50,
        margin: 0,
        padding: 0,
    },
    image: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        justifyContent: 'space-between'
    },
    text: {
        color: '#777777',
        fontSize: 28,
        lineHeight: 84,
        fontWeight: 'bold',
        textAlign: 'center',
    },
    text2: {
        color: '#444444',
        fontSize: 15,
        lineHeight: 84,
        fontWeight: 'bold',
        textAlign: 'center',
    },
});
