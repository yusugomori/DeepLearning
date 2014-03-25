#!/usr/bin/perl -w

use strict;
use warnings;
use Data::Dumper;

package LogisticRegression;

sub new {
  my $pkg = shift;
  my $self = {
              N => undef,
              n_in => undef,
              n_out => undef,
              @_,
             };

  bless $self, $pkg;

  my $i;
  my $j;

  my $W = [];
  my $b = [];
  for($i=0; $i<$self->{n_out}; $i++) {
    $$W[$i] = [];
    for($j=0; $j<$self->{n_in}; $j++) {
      $$W[$i][$j] = 0;
    }
    $$b[$i] = 0;
  }
  $self->{W} = $W;
  $self->{b} = $b;

  return $self;
}

sub train {
  my $self = shift;
  my ($x, $y, $lr) = @_;

  my $i;
  my $j;
  my $p_y_given_x = [];
  my $dy = [];

  for($i=0; $i<$self->{n_out}; $i++) {
    $$p_y_given_x[$i] = 0;
    for($j=0; $j<$self->{n_in}; $j++) {
      $$p_y_given_x[$i] += $self->{W}[$i][$j] * $$x[$j];
    }
    $$p_y_given_x[$i] += $self->{b}[$i];
  }
  $self->softmax($p_y_given_x);

  for($i=0; $i<$self->{n_out}; $i++) {
    $$dy[$i] = $$y[$i] - $$p_y_given_x[$i];

    for($j=0; $j<$self->{n_in}; $j++) {
      $self->{W}[$i][$j] += $lr * $$dy[$i] * $$x[$i] / $self->{N};
    }

    $self->{b}[$i] += $lr * $$dy[$i] / $self->{N};
  }
}


sub softmax {
  my $self = shift;
  my ($x) = @_;

  my $i;
  my $max = 0.0;
  my $sum = 0.0;

  for($i=0; $i<$self->{n_out}; $i++) {
    $max = $$x[$i] if($max < $$x[$i]);
  }

  for($i=0; $i<$self->{n_out}; $i++) {
    $$x[$i] = exp($$x[$i] - $max);
    $sum += $$x[$i];
  }

  for($i=0; $i<$self->{n_out}; $i++) {
    $$x[$i] /= $sum;
  }
}

sub predict {
  my $self = shift;
  my ($x, $y) = @_;

  for(my $i=0; $i<$self->{n_out}; $i++) {
    $$y[$i] = 0;

    for(my $j=0; $j<$self->{n_in}; $j++) {
      $$y[$i] += $self->{W}[$i][$j] * $$x[$j];
    }
    $$y[$i] += $self->{b}[$i];
  }

  $self->softmax($y);
}



1;



sub test_lr {
  my $i;
  my $j;
  my $epoch;

  my $learning_rate = 0.1;
  my $n_epochs = 500;

  my $train_N = 6;
  my $test_N = 3;
  my $n_in = 6;
  my $n_out = 2;

  # training data
  my $train_X = [
                 [1, 1, 1, 0, 0, 0],
                 [1, 0, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [0, 0, 1, 1, 1, 0],
                 [0, 0, 1, 1, 0, 0],
                 [0, 0, 1, 1, 1, 0],
                ];
  my $train_Y = [
                 [1, 0],
                 [1, 0],
                 [1, 0],
                 [0, 1],
                 [0, 1],
                 [0, 1]
                ];

  # construct
  my $classifier = LogisticRegression->new(N=>$train_N, n_in=>$n_in, n_out=>$n_out);


  # train
  for($epoch=0; $epoch<$n_epochs; $epoch++) {
    for($i=0; $i<$train_N; $i++) {
      $classifier->train($$train_X[$i], $$train_Y[$i], $learning_rate);
    }
    # $learning_rate *= 0.95;
  }


  # test data
  my $test_X = [
                [1, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 0],
                [1, 1, 1, 1, 1, 0],
               ];

  my $test_Y = [];
  for($i=0; $i<$test_N; $i++) {
    $$test_Y[$i] = [];
    for($j=0; $j<$n_out; $j++) {
      $$test_Y[$i][$j] = 0;
    }
  }

  # test
  for($i=0; $i<$test_N; $i++) {
    $classifier->predict($$test_X[$i], $$test_Y[$i]);

    for($j=0; $j<$n_out; $j++) {
      printf("%.5f ", $$test_Y[$i][$j]);
    }
    print "\n";
  }
}



test_lr();
