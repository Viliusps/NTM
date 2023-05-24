import {Dropzone, IMAGE_MIME_TYPE} from "@mantine/dropzone";
import {Group, Text, Image, useMantineTheme, rem, Button, NumberInput, Box, Container} from '@mantine/core';
import { IconUpload, IconPhoto, IconX } from '@tabler/icons-react';
import {YearPickerInput} from "@mantine/dates";
import {useRef, useState} from "react";
import axios from "axios";

function App() {
    const [makeDate, setMakeDate] = useState(null);
    const [kilometrage, setKilometrage] = useState(0);
    const [engine, setEngine] = useState(0);
    const [uploadedFile, setUploadedFile] = useState([]);
    const [prediction, setPrediction] = useState('');
    const theme = useMantineTheme();
    const formRef = useRef();

    const handleSubmit = async () => {
        await axios.post('http://localhost:5000/predict', {
            image: uploadedFile[0],
            kilometrage: kilometrage,
            make_date: makeDate,
            engine: engine
        },  {
            headers: {
                'Content-Type': 'multipart/form-data',
            }})
            .then((e) => {
            setPrediction(e.data.prediction.toFixed(0));
        })
    }

    return (
        <form ref={formRef}>
            <Container size={420} my={40}>
                <Box maw={320} mx="auto">
                    <YearPickerInput
                        label="Pagaminimo metai"
                        placeholder="Pagaminimo metai"
                        value={makeDate}
                        onChange={setMakeDate}
                        mx="auto"
                        maw={400}
                        required
                    />
                    <NumberInput min={0} label="Rida" value={kilometrage} onChange={setKilometrage} required/>
                    <NumberInput min={0} label="Galingumas" value={engine} onChange={setEngine} required/>
                    <br/>
                    <Dropzone
                        onDrop={setUploadedFile}
                        onReject={(files) => console.log('rejected files', files)}
                        maxSize={3 * 1024 ** 2}
                        accept={IMAGE_MIME_TYPE}
                        aria-required
                        maxFiles={1}
                    >
                        {uploadedFile.length>0?<Image src={URL.createObjectURL(uploadedFile[0])}/>:
                        <Group position="center" spacing="xl" style={{ minHeight: rem(220), pointerEvents: 'none' }}>
                            <Dropzone.Accept>
                                <IconUpload
                                    size="3.2rem"
                                    stroke={1.5}
                                    color={theme.colors[theme.primaryColor][theme.colorScheme === 'dark' ? 4 : 6]}
                                />
                            </Dropzone.Accept>
                            <Dropzone.Reject>
                                <IconX
                                    size="3.2rem"
                                    stroke={1.5}
                                    color={theme.colors.red[theme.colorScheme === 'dark' ? 4 : 6]}
                                />
                            </Dropzone.Reject>
                            <Dropzone.Idle>
                                <IconPhoto size="3.2rem" stroke={1.5} />
                            </Dropzone.Idle>

                            <div>
                                <Text size="xl" inline>
                                    Drag image here or click to select the file
                                </Text>
                                <Text size="sm" color="dimmed" inline mt={7}>
                                    The file should not exceed 5mb
                                </Text>
                            </div>
                        </Group>}
                    </Dropzone>
                    {prediction?<Text>Rekomendacija: {prediction}€</Text>:null}
                    <Group position="center" mt="xl">
                        <Button onClick={() => (formRef.current.reportValidity() ? handleSubmit() : null)}>
                            Submit</Button>
                    </Group>
                </Box>
            </Container>
        </form>
    );
}

export default App;
