# Cyclic Scheduling of Stream

A custom fitness evaluation for Stream to perform row-wise (or close to row-wise) layer fusion scheduling in Stream.
This library exploits the very symmetric properties of row-wise layer fusion to speed up analysis time and explore further schedulings.  
The drawback of this technique is a small loss at the prologue and the epilogue in terms of latency.

![scheduling](img.png)

## NOTE
This library uses a custom version of Stream that allows custom fitness evaluation at [stream-fork](https://github.com/Andful/stream-fork/tree/draft-custom-fitness-evaluation)

## USAGE
```bash
pip3 install .
jupyter notebook ./example
```
