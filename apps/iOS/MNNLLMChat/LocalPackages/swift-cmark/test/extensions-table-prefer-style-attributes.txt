---
title: Extensions test with --table-prefer-style-attributes
author: FUJI Goro
version: 0.1
date: '2018-02-20'
license: '[CC-BY-SA 4.0](http://creativecommons.org/licenses/by-sa/4.0/)'
...

## Tables

Table alignment:

```````````````````````````````` example
aaa | bbb | ccc | ddd | eee
:-- | --- | :-: | --- | --:
fff | ggg | hhh | iii | jjj
.
<table>
<thead>
<tr>
<th style="text-align: left">aaa</th>
<th>bbb</th>
<th style="text-align: center">ccc</th>
<th>ddd</th>
<th style="text-align: right">eee</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left">fff</td>
<td>ggg</td>
<td style="text-align: center">hhh</td>
<td>iii</td>
<td style="text-align: right">jjj</td>
</tr>
</tbody>
</table>
````````````````````````````````
