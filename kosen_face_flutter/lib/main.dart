import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;


void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget{
  const MyApp({Key? key}) : super (key: key);
  @override
  Widget build(BuildContext context){
    return MaterialApp(
      title: 'Kosen Face App',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home:  MyHomePage(),
    );
  }
}

class MyHomePage extends StatefulWidget {
  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  String _responseText = 'サーバーからの応答がここに表示されます';

  Future<void> _sendRequest() async {
    final url = Uri.parse('http://localhost:8000');
    final response = await http.get(url);
    setState(() {
      _responseText = response.body;
    });
  }

  Future<void> _sendPostRequest() async {
    final url = Uri.parse('http://localhost:8000');
    final response = await http.post(url, body: {'key': 'value'});
    setState(() {
      _responseText = response.body;
    });
  }
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Kosen Face App'),
      ),
      body: Padding(
        padding: EdgeInsets.all(16.0),
        child: Column(
          children: [
            Text(
              _responseText,
              style: TextStyle(fontSize: 16),
            ),
            SizedBox(height: 20),
            ElevatedButton(
              onPressed: _sendRequest,
              child: Text('GET Request'),
            ),
            SizedBox(height: 10),
            ElevatedButton(
              onPressed: _sendPostRequest,
              child: Text('POST Request'),
            ),
          ],
        ),
      ),
    );
  }
}