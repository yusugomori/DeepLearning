#!/usr/bin/env ruby
# -*- coding: utf-8 -*-

class LogisticRegression
  def initialize(n, n_in, n_out)
    @N = n
    @n_in = n_in
    @n_out = n_out

    @W = Array.new(@n_out).map{ Array.new(@n_in, 0) }
    @b = Array.new(@n_out, 0)
  end

  def train(x, y, lr)
    p_y_given_x = Array.new(@n_out, 0)
    dy = Array.new(@n_out, 0)

    for i in 0...@n_out
      for j in 0...@n_in
        p_y_given_x[i] += @W[i][j].to_f * x[j]
      end
      p_y_given_x[i] += @b[i]
    end
    softmax(p_y_given_x)

    for i in 0...@n_out
      dy[i] = y[i].to_f - p_y_given_x[i]

      for j in 0...@n_in
        @W[i][j] += lr * dy[i] * x[j] / @N
      end

      @b[i] += lr * dy[i] / @N
    end
  end

  def softmax(x)
    max = 0.0
    sum = 0.0

    for i in 0...@n_out
      if max < x[i]
        max = x[i]
      end
    end

    for i in 0...@n_out
      x[i] = Math.exp(x[i] - max)
      sum += x[i]
    end

    for i in 0...@n_out
      x[i] /= sum
    end
  end

  def predict(x, y)
    for i in 0...@n_out
      y[i] = 0
      for j in 0...@n_in
        y[i] += @W[i][j] * x[j]
      end
      y[i] += @b[i]
    end

    softmax(y)
  end
  
end



def test_lr
  learning_rate = 0.1
  n_epochs = 500

  train_N = 6
  test_N = 3
  n_in = 6
  n_out = 2

  train_X = [
             [1, 1, 1, 0, 0, 0],
             [1, 0, 1, 0, 0, 0],
             [1, 1, 1, 0, 0, 0],
             [0, 0, 1, 1, 1, 0],
             [0, 0, 1, 1, 0, 0],
             [0, 0, 1, 1, 1, 0]
            ]

  train_Y = [
             [1, 0],
             [1, 0],
             [1, 0],
             [0, 1],
             [0, 1],
             [0, 1]
            ]

  # construct
  classifier = LogisticRegression.new(train_N, n_in, n_out)

  # train
  for epoch in 0...n_epochs
    for i in 0...train_N
      classifier.train(train_X[i], train_N[i], learning_rate)
    end
    # learning_rate *= 0.95
  end

  # test data
  test_X = [
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0],
            [1, 1, 1, 1, 1, 0],
           ]

  test_Y = Array.new(test_N).map{ Array.new(n_out, 0) }


  # test
  for i in 0...test_N
    classifier.predict(test_X[i], test_Y[i])
    for j in 0...n_out
      printf "%.5f ", test_Y[i][j]
    end
    puts
  end
  
end


if __FILE__ == $0
  test_lr()
end
