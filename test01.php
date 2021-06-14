<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>備忘録</title>
  <link href="https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css" rel="stylesheet" integrity="sha384-wvfXpqpZZVQGK6TAh5PVlGOfQNHSoD2xbE+QkPxCAFlNEevoEH3Sl0sibVcOQVnN" crossorigin="anonymous">
  <link rel="stylesheet" type="text/css" href="stylesheet.css">
</head>
<body>

<?php
// ファイルを書き込み専用でオープンする
$fp = fopen('write.txt', 'w');
// ファイルに文字列を書き込む
fputs($fp, "１行目\n");
fputs($fp, "２行目\n");
fputs($fp, "３行目\n");
// ファイルをクローズする
fclose($fp);
?>

</body>
</html>