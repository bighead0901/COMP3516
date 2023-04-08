import { StatusBar } from 'expo-status-bar';
import * as React from 'react';
import { StyleSheet, View, ImageBackground } from 'react-native';
import { BottomNavigation, Text, IconButton, MD3Colors } from 'react-native-paper';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { Audio } from 'expo-av';
import { API } from 'aws-amplify';

const RecordRoute = () => {
    const [recording, setRecording] = React.useState();
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
            await Audio.requestPermissionsAsync();
            await Audio.setAudioModeAsync({
                allowsRecordingIOS: true,
                playsInSilentModeIOS: true,
            });

            console.log('Starting recording..');
            const { recording } = await Audio.Recording.createAsync(
                Audio.RecordingOptionsPresets.HIGH_QUALITY,
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
        console.log('Stopping recording..' + recordingDuration);
        setRecording(undefined);
        await recording.stopAndUnloadAsync();
        await Audio.setAudioModeAsync({
          allowsRecordingIOS: false,
        });
        const uri = recording.getURI();
        console.log('Recording stopped and stored at', uri);
        const response = await API.get('comp3516api', '/');
        console.log(response)
    }

    return(
        <ImageBackground source={require('./src/bg.png')} resizeMode="cover" style={styles.image}>
            <View style={styles.container}>
                <Text style={styles.text}>Speech Liveness Detection</Text>
                <Text style={styles.text2}>
                    {recording ? `Recording... ${recordingDuration}` : 'Press button below to start recording'}
                </Text>
            </View>
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
