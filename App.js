import { StatusBar } from 'expo-status-bar';
import * as React from 'react';
import { StyleSheet, View, ImageBackground } from 'react-native';
import { BottomNavigation, Text } from 'react-native-paper';
import { SafeAreaProvider } from 'react-native-safe-area-context';

export default function App() {
    const RecordRoute = () => {
        return(
            <ImageBackground source={require('./src/bg.png')} resizeMode="cover" style={styles.image}>
                <Text style={styles.text}>Speech Liveness Detection</Text>
            </ImageBackground>
        )
    }

    const FilesRoute = () => {
        return(
            <Text>History</Text>
        )
    }

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
            <BottomNavigation
                navigationState={{ index, routes }}
                onIndexChange={setIndex}
                renderScene={renderScene}
            />
        </SafeAreaProvider>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        width: "500",
        height: "100%",
        backgroundColor: '#fff',
        alignItems: 'center',
        justifyContent: 'center',
    },
    image: {
        flex: 1,
        justifyContent: 'center',
    },
    text: {
        color: '#777777',
        fontSize: 23,
        lineHeight: 84,
        fontWeight: 'bold',
        textAlign: 'center',
    },
});
